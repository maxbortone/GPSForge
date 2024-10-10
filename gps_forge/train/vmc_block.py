import os
import time
import ml_collections
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import GPSKet as qk
from copy import deepcopy
from absl import logging
from netket.utils.types import Callable, PyTree, Array
from gps_forge.systems import get_system
from gps_forge.models import get_model
from gps_forge.samplers import get_sampler
from VMCutils import Timer, CSVLogger
from VMCutils import restore_best_params, save_best_params
from netket.utils.mpi import rank as mpi_rank


def get_apply_fun_block(vstate: nk.vqs.MCState) -> Callable:
    apply_fun = vstate._apply_fun
    def apply_fun_block(variables: PyTree, x: Array):
        block = variables['params']['block']
        idx_block = variables['cache']['idx_block']
        model_params = variables['cache']['model_params']
        model_params, unravel_fun = nk.jax.tree_ravel(model_params)
        model_params = model_params.at[idx_block].set(block)
        model_params = unravel_fun(model_params)
        model_state = variables['orbitals']
        return apply_fun({'params': model_params, 'orbitals': model_state}, x)
    return apply_fun_block

def get_init_fun_block(vstate: nk.vqs.MCState, block_size: int) -> Callable:
    init_fun = vstate._init_fun
    def init_fun_block(args: PyTree, dummy_input: Array):
        key = args['params']
        key_b, _ = jax.random.split(key)
        model_variables = init_fun(key, dummy_input)
        n_params = nk.jax.tree_size(model_variables['params'])
        idx_block = jax.random.randint(key_b, (block_size,), 0, n_params)
        model_params = model_variables['params']
        block = nk.jax.tree_ravel(model_params)[0][idx_block]
        block_variables = deepcopy(model_variables)
        block_variables['params'] = {'block': block}
        block_variables['cache'] = {'idx_block': idx_block, 'model_params': model_params}
        return block_variables
    return init_fun_block

def get_update_block_fun(vstate: nk.vqs.MCState, vstate_block: nk.vqs.MCState, qgt: qk.optimizer.qgt.QGTJacobianDenseRMSProp, solver: Callable, config: ml_collections.ConfigDict) -> Callable:
    _, unravel_fun = nk.jax.tree_ravel(vstate.parameters.unfreeze())
    def update_block(carry, x):
        dp, model_params, samples = carry
        idx_block, grad_block, ema_block = x
        block = jnp.take(model_params, idx_block)
        vstate_block.parameters = {'block': block}
        vstate_block.model_state = {'cache': {'idx_block': idx_block, 'model_params': unravel_fun(model_params)}, **vstate.model_state}
        vstate_block._samples = samples
        lhs = qgt(vstate_block, ema_block, diag_shift=config.optimizer.diag_shift, eps=config.optimizer.eps, mode=config.optimizer.mode)
        dp_block, _ = lhs.solve(solver, grad_block)
        dp = dp.at[idx_block].set(dp_block)
        return (dp, model_params, samples), None
    return update_block

def get_idx_blocks(ema: Array, n_blocks: int):
    sort_ema = jnp.argsort(ema)
    idx_blocks = jnp.array_split(sort_ema, n_blocks) # (n_blocks, block_size)
    return jnp.array(idx_blocks)

def get_update_ema_fun(config: ml_collections.ConfigDict) -> Callable:
    def update_ema(nu, g, step):
        if jnp.iscomplexobj(g):
            # This assumes that the parameters are split into complex and real parts later on (done in the QGT implementation)
            squared_g = (g.real**2 + 1.j * g.imag**2)
        else:
            squared_g = (g**2)
        ema = config.optimizer.decay*nu + (1-config.optimizer.decay)*squared_g
        return ema/(1-config.optimizer.decay**step)
    return update_ema

def vmc_block(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using VMC and a blocked SR optimizer."""

    # Setup system
    ha = get_system(config, workdir=workdir)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config, hi, g, ha, workdir)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Quantum geometric tensor
    solver = jax.scipy.sparse.linalg.cg
    qgt = qk.optimizer.qgt.QGTJacobianDenseRMSProp

    # Variational state
    vs = nk.vqs.MCState(sa, ma, **config.variational_state)
    vs.n_samples = config.variational_state.n_samples * config.optimizer.n_samples_multiplier

    # Block variational state
    assert vs.n_parameters % config.optimizer.n_blocks == 0
    block_size = vs.n_parameters // config.optimizer.n_blocks
    vs_block = deepcopy(vs)
    vs_block._apply_fun = get_apply_fun_block(vs)
    vs_block._init_fun = get_init_fun_block(vs, block_size)
    vs_block.init()
    ema = jnp.zeros(nk.jax.tree_size(vs.parameters))
    update_ema = get_update_ema_fun(config)
    update_block = get_update_block_fun(vs, vs_block, qgt, solver, config)

    # Logger
    if mpi_rank == 0:
        fieldnames = list(nk.stats.Stats().to_dict().keys())+["Runtime"]
        logger = CSVLogger(os.path.join(workdir, "metrics.csv"), fieldnames)

    # Run training loop
    if mpi_rank == 0:
        logging.info('Starting training loop; initial compile can take a while...')
        timer = Timer(config.total_steps)
        t0 = time.time()
        best_params = restore_best_params(workdir)
        best_energy = best_params["Energy"] if best_params else np.inf
        best_variance = best_params["Variance"] if best_params else np.inf
    initial_step = 1
    total_steps = config.total_steps
    for step in range(initial_step, total_steps + 1):
        # Sample configurations
        samples = vs.sample()
        samples_en = samples[:, ::config.optimizer.n_samples_multiplier, :]

        # Compute energy and gradient
        vs._samples = samples_en
        energy, grad = vs.expect_and_grad(ha)
        grad, unravel_fun = nk.jax.tree_ravel(grad)

        # Update exponential moving average (EMA)
        ema = update_ema(ema, grad, step)

        # Compute block indices
        idx_blocks = get_idx_blocks(ema, config.optimizer.n_blocks)
        grad_blocks = jax.vmap(lambda idx: jnp.take(grad, idx))(idx_blocks)
        ema_blocks = jax.vmap(lambda idx: jnp.take(ema, idx))(idx_blocks)

        # Update block parameters
        model_params, _ = nk.jax.tree_ravel(vs.parameters.unfreeze())
        dp = jnp.zeros_like(grad)
        (dp, _, _), _ = jax.lax.scan(update_block, (dp, model_params, samples), (idx_blocks, grad_blocks, ema_blocks))

        # Update parameters
        model_params = model_params - config.optimizer.learning_rate * dp
        vs.parameters = unravel_fun(model_params)

        # Report compilation time
        if mpi_rank == 0 and step == initial_step:
            logging.info(f"First step took {time.time() - t0:.1f} seconds.")

        # Update timer
        if mpi_rank == 0:
            timer.update(step)

        # Log data
        if mpi_rank == 0:
            logger(step, {**energy.to_dict(), "Runtime": timer.runtime})

        # Save best energy params
        if mpi_rank == 0 and energy.mean.real < best_energy and energy.variance < best_variance:
            best_energy = energy.mean.real
            best_variance = energy.variance
            save_best_params(workdir, {"Energy": best_energy, "Variance": best_variance, "Parameters": vs.parameters})
            logging.info(f"Stored best parameters at step {step} with energy {energy}")

        # Report training metrics
        if mpi_rank == 0 and config.progress_every and step % config.progress_every == 0:
            grad_norm = np.linalg.norm(grad)
            done = step / total_steps
            logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                         f"E: {energy}, "
                         f"||∇E||: {grad_norm:.4f}, "
                         f"{timer}")

    return 