import os
import ml_collections
import jax
import numpy as np
import netket as nk
from netket.experimental.driver import VMC_SRt
import GPSKet as qk
import jax.numpy as jnp
import pandas as pd
import plotext as plt
from absl import logging
from timeit import default_timer as timer
from datetime import timedelta
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from ar_qgps.samplers import get_sampler
from ar_qgps.variational_states import get_variational_state
from ar_qgps.optimizers import get_optimizer
from VMCutils import MPIVars, CSVLogger


# Function that times the execution of a JAX function over many MPI processes
def timeit(func, *args, repeat=1):
    runtimes = []
    for _ in range(repeat):
        # Time function execution
        start = timer()
        result = func(*args)
        jax.tree_map(lambda x: x.block_until_ready(), result)
        end = timer()
        runtime = timedelta(seconds=end - start)
        # Get runtime from all MPI processes
        runtimes.append(MPIVars.comm.allgather(runtime.total_seconds()))
    return result, np.mean(runtimes)

def benchmark(config: ml_collections.ConfigDict, workdir: str):
    """Benchmark a VMC Ansatz on a system"""

    # Setup system
    ha = get_system(config, workdir)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config, hi, g, ha, workdir)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Variational state
    vs = get_variational_state(config, ma, hi, sa)

    # Optimizer
    op, sr = get_optimizer(config, vs)

    # Driver
    if config.optimizer_name == 'minSR':
        solver = lambda A, b: jnp.linalg.lstsq(A, b, rcond=config.optimizer.rcond)[0]
        vmc = qk.driver.minSRVMC(ha, op, variational_state=vs, mode=config.optimizer.mode, minSR_solver=solver)
    elif config.optimizer_name == 'kernelSR':
        vmc = VMC_SRt(ha, op, variational_state=vs, jacobian_mode=config.optimizer.mode, diag_shift=config.optimizer.diag_shift)
    else:
        vmc = nk.driver.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Logger
    if MPIVars.rank == 0:
        fieldnames = ["Component", "Runtime"]
        logger = CSVLogger(os.path.join(workdir, "runtimes.csv"), fieldnames)
        total_steps = 6 if sr is not None else 5

    # Benchmark compilation
    # NOTE: this could in theory be done without evaluating everything twice via
    # ```
    # timeit(f.lower(*args).compile())
    # ```
    # (see: https://github.com/google/jax/discussions/9716)
    def advance(vs, sr):
        vs.reset()
        samples = vs.sample()
        samples = samples.reshape((-1, samples.shape[-1]))
        vs.log_value(samples)
        energy, grad = vs.expect_and_grad(ha)
        if sr is not None:
            sr(vs, grad, 1)
        return

    if MPIVars.rank == 0:
        step = 1
        logging.info(f"[{step}/{total_steps}] Benchmarking compilation...")
    _, runtime_uncompiled = timeit(advance, vs, sr, repeat=config.repeat)
    _, runtime_compiled = timeit(advance, vs, sr, repeat=config.repeat)
    runtime = runtime_uncompiled - runtime_compiled
    if MPIVars.rank == 0:
        logger(step, {"Component": "Compilation", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark sampling
    vs.reset()
    if MPIVars.rank == 0:
        step += 1
        logging.info(f"[{step}/{total_steps}] Benchmarking sampling...")
    samples, runtime = timeit(vs.sample, repeat=config.repeat)
    samples = samples.reshape((-1, samples.shape[-1]))
    if MPIVars.rank == 0:
        logger(step, {"Component": "Sampling", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark amplitude evaluation
    if MPIVars.rank == 0:
        step += 1
        logging.info(f"[{step}/{total_steps}] Benchmarking amplitude evaluation...")
    # samples = jnp.squeeze(samples)
    _, runtime = timeit(vs.log_value, samples, repeat=config.repeat)
    if MPIVars.rank == 0:
        logger(step, {"Component": "Amplitude", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark energy and gradient evaluation
    if MPIVars.rank == 0:
        step += 1
        logging.info(f"[{step}/{total_steps}] Benchmarking energy and gradient evaluation...")
    (energy, grad), runtime = timeit(vs.expect_and_grad, ha, repeat=config.repeat)
    if MPIVars.rank == 0:
        logger(step, {"Component": "Energy", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark preconditioner evaluation
    if sr is not None:
        if MPIVars.rank == 0:
            step += 1
            logging.info(f"[{step}/{total_steps}] Benchmarking preconditioner evaluation...")
        _, runtime = timeit(sr, vs, grad, 1, repeat=config.repeat)
        if MPIVars.rank == 0:
            logger(step, {"Component": "Preconditioner", "Runtime": runtime})
            logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark optimization step
    vs.reset()
    if MPIVars.rank == 0:
        step += 1
        logging.info(f"[{step}/{total_steps}] Benchmarking optimization step...")
    _, runtime = timeit(vmc.advance, 1, repeat=config.repeat)
    if MPIVars.rank == 0:
        logger(step, {"Component": "Optimization", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Plot to terminal
    df = pd.read_csv(os.path.join(workdir, "runtimes.csv"), header=0, delimiter=",")
    plt.simple_bar(df["Component"], df["Runtime"], width=100, title="Runtime (s)")
    plt.show()

    return 