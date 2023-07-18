import os
import time
import ml_collections
import jax
import numpy as np
import netket as nk
from copy import deepcopy
from absl import logging
from netket.utils.types import Callable, PyTree, Array
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from ar_qgps.samplers import get_sampler
from VMCutils import MPIVars, Timer, CSVLogger
from VMCutils import restore_best_params, save_best_params


@jax.jit
def solve(A, b):
    return jax.scipy.sparse.linalg.cg(A, b)[0]

def get_apply_fun_block(vstate: nk.vqs.MCState) -> Callable:
    def apply_fun_block(variables: PyTree, x: Array):
        block = variables['params']['block']
        block_idx = variables['block_idx']
        model_params = variables['model_params']
        model_params, unravel_fun = nk.jax.tree_ravel(model_params)
        model_params = model_params.at[block_idx].set(block)
        model_params = unravel_fun(model_params)
        return vstate._apply_fun({'params': model_params}, x)
    return apply_fun_block

def get_init_fun_block(vstate: nk.vqs.MCState, block_size: int) -> Callable:
    def init_fun_block(args: PyTree, dummy_input: Array):
        key = args['params']
        key_b, _ = jax.random.split(key)
        model_variables = vstate._init_fun(key, dummy_input)
        n_params = nk.jax.tree_size(model_variables['params'])
        idx = jax.random.randint(key_b, (block_size,), 0, n_params)
        model_params = model_variables['params']
        block = nk.jax.tree_ravel(model_params)[0][idx]
        variables = {}
        variables['params'] = {'block': block}
        variables['block_idx'] = idx
        variables['model_params'] = model_params
        return variables
    return init_fun_block

def vmc_block(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using VMC and a blocked SR optimizer."""

    # Setup system
    ha = get_system(config, workdir=workdir)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config, hi, g, ha)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Variational state
    vs = nk.vqs.MCState(sa, ma, **config.variational_state)

    # Block variational state
    vs_block = deepcopy(vs)
    vs_block._apply_fun = get_apply_fun_block(vs)
    vs_block._init_fun = get_init_fun_block(vs, config.optimizer.block_size)
    vs_block.init()

    # Quantum geometric tensor
    qgt = nk.optimizer.qgt.QGTJacobianDense(mode=config.optimizer.mode)

    # Logger
    if MPIVars.rank == 0:
        fieldnames = list(nk.stats.Stats().to_dict().keys())+["Runtime"]
        logger = CSVLogger(os.path.join(workdir, "metrics.csv"), fieldnames)

    # Run training loop
    if MPIVars.rank == 0:
        logging.info('Starting training loop; initial compile can take a while...')
        timer = Timer(config.total_steps)
        t0 = time.time()
        best_params = restore_best_params(workdir)
        best_energy = best_params["Energy"] if best_params else np.inf
        best_variance = best_params["Variance"] if best_params else np.inf
    initial_step = 1
    total_steps = config.total_steps
    for step in range(initial_step, total_steps + 1):
        # Compute energy and gradient
        energy, grad = vs.expect_and_grad(ha)
        grad, unravel_fun = nk.jax.tree_ravel(grad)

        # Find parameters with largest gradient
        model_params, unravel_fun = nk.jax.tree_ravel(vs.parameters.unfreeze())
        block_idx = np.argsort(np.linalg.norm(grad))[-config.optimizer.block_size:]
        block = model_params[block_idx]
        vs_block.parameters = {'block': block}
        vs_block.model_state = {'block_idx': block_idx, 'model_params': unravel_fun(model_params)}

        # Find SR update of block of parameters
        vs_block._samples = vs.samples
        S_block = qgt(vs_block, diag_shift=config.optimizer.diag_shift).to_dense()
        grad_block = grad[block_idx]
        dp_block = solve(S_block, grad_block)

        # Combine block update with other parameters
        dp = grad
        dp = dp.at[block_idx].set(dp_block)

        # Update parameters
        model_params = model_params - config.optimizer.learning_rate * dp
        vs.parameters = unravel_fun(model_params)

        # Report compilation time
        if MPIVars.rank == 0 and step == initial_step:
            logging.info(f"First step took {time.time() - t0:.1f} seconds.")

        # Update timer
        if MPIVars.rank == 0:
            timer.update(step)

        # Log data
        if MPIVars.rank == 0:
            logger(step, {**energy.to_dict(), "Runtime": timer.runtime})

        # Save best energy params
        if MPIVars.rank == 0 and energy.mean.real < best_energy and energy.variance < best_variance:
            best_energy = energy.mean.real
            best_variance = energy.variance
            save_best_params(workdir, {"Energy": best_energy, "Variance": best_variance, "Parameters": vs.parameters})
            logging.info(f"Stored best parameters at step {step} with energy {energy}")

        # Report training metrics
        if MPIVars.rank == 0 and config.progress_every and step % config.progress_every == 0:
            grad_norm = np.linalg.norm(grad)
            done = step / total_steps
            logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                         f"E: {energy}, "
                         f"||∇E||: {grad_norm:.4f}, "
                         f"{timer}")

    return 