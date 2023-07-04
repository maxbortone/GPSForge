import os
import ml_collections
import jax
import numpy as np
import netket as nk
import GPSKet as qk
import jax.numpy as jnp
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
    ha = get_system(config)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config, hi, g, ha)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Variational state
    vs = get_variational_state(config, ma, hi, sa)

    # Optimizer
    op, sr = get_optimizer(config, vs)

    # Driver
    if config.optimizer_name == 'minSR':
        solver = lambda A, b: jnp.linalg.lstsq(A, b, rcond=config.optimizer.rcond)[0]
        vmc = qk.driver.minSRVMC(ha, op, variational_state=vs, mode=config.optimizer.mode, minSR_solver=solver, diag_shift=config.optimizer.diag_shift)
    else:
        vmc = nk.driver.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Logger
    if MPIVars.rank == 0:
        fieldnames = ["Component", "Runtime"]
        logger = CSVLogger(os.path.join(workdir, "runtimes.csv"), fieldnames)

    # Benchmark compilation
    if MPIVars.rank == 0:
        logging.info("[0/4] Benchmarking compilation...")
    _, runtime = timeit(vmc.advance, 1, repeat=config.repeat)
    if MPIVars.rank == 0:
        logger(0, {"Component": "Compilation", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark sampling
    _ = vs.sample()
    if MPIVars.rank == 0:
        logging.info("[1/4] Benchmarking sampling...")
    samples, runtime = timeit(vs.sample, repeat=config.repeat)
    samples = samples.reshape((-1, samples.shape[-1]))
    if MPIVars.rank == 0:
        logger(1, {"Component": "Sampling", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark amplitude evaluation
    _ = vs.log_value(samples)
    if MPIVars.rank == 0:
        logging.info("[2/4] Benchmarking amplitude evaluation...")
    # samples = jnp.squeeze(samples)
    _, runtime = timeit(vs.log_value, samples, repeat=config.repeat)
    if MPIVars.rank == 0:
        logger(2, {"Component": "Amplitude", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark local energy evaluation
    _ = vs.local_estimators(ha)
    if MPIVars.rank == 0:
        logging.info("[3/4] Benchmarking local energy evaluation...")
    _, runtime = timeit(vs.local_estimators, ha, repeat=config.repeat)
    if MPIVars.rank == 0:
        logger(3, {"Component": "Local energy", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    # Benchmark optimization step
    if MPIVars.rank == 0:
        logging.info("[4/4] Benchmarking optimization step...")
    _, runtime = timeit(vmc._forward_and_backward, repeat=config.repeat)
    if MPIVars.rank == 0:
        logger(4, {"Component": "Optimization", "Runtime": runtime})
        logging.info(f"Done! This took {runtime} seconds.")

    return 