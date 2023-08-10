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
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from ar_qgps.samplers import get_sampler
from VMCutils import MPIVars, Timer, CSVLogger
from VMCutils import restore_best_params, save_best_params


def get_block_idx(config: ml_collections.ConfigDict, hilbert: qk.hilbert.FermionicDiscreteHilbert) -> PyTree:
    M = config.model.M
    D = hilbert._local_size
    norb = hilbert.size
    nelec = np.sum(hilbert._n_elec)
    total_bond_dim = M*norb*nelec
    block_idx = []
    for k in range(norb):
        idx = []
        for n in range(D):
            for i in range(nelec):
                for m in range(M):
                    for l in range(norb):
                        idx.append(n*total_bond_dim+(i*nelec+m)*norb+l+k*norb*norb)
        idx = np.array(idx, np.int32)
        block_idx.append(idx)
    block_idx = jnp.array(block_idx, np.int32)
    return block_idx

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
    ma = get_model(config, hi, g, ha, workdir)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Quantum geometric tensor
    solver = jax.scipy.sparse.linalg.cg
    qgt = qk.optimizer.qgt.QGTJacobianDenseRMSProp

    # Variational state
    vs = nk.vqs.MCState(sa, ma, **config.variational_state)
    _, unravel_fun = nk.jax.tree_ravel(vs.parameters.unfreeze())

    # Block variational state
    block_size = vs.n_parameters // hi.size
    vs_block = deepcopy(vs)
    vs_block._apply_fun = get_apply_fun_block(vs)
    vs_block._init_fun = get_init_fun_block(vs, block_size)
    vs_block.init()
    block_idx = get_block_idx(config, hi)
    ema = jnp.zeros(nk.jax.tree_size(vs.parameters))
    def update_ema(nu, g, step):
        if jnp.iscomplexobj(g):
            # This assumes that the parameters are split into complex and real parts later on (done in the QGT implementation)
            squared_g = (g.real**2 + 1.j * g.imag**2)
        else:
            squared_g = (g**2)
        ema = config.optimizer.decay*nu + (1-config.optimizer.decay)*squared_g
        return ema/(1-config.optimizer.decay**step)
    def update_block(model_params, samples, idx, g, e):
        block = jnp.take(model_params, idx)
        vs_block.parameters = {'block': block}
        vs_block.model_state = {'block_idx': idx, 'model_params': unravel_fun(model_params)}
        vs_block._samples = samples
        lhs = qgt(vs_block, e, diag_shift=config.optimizer.diag_shift, eps=config.optimizer.eps, mode=config.optimizer.mode)
        dp_block, _ = lhs.solve(solver, g)
        return dp_block
    update_blocks = jax.jit(jax.vmap(update_block, in_axes=(None, None, 0, 0, 0)))

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
        grad_block = jax.vmap(lambda idx: jnp.take(grad, idx))(block_idx)

        # Update exponential moving average (EMA)
        ema = update_ema(ema, grad, step)
        ema_block = jax.vmap(lambda idx: jnp.take(ema, idx))(block_idx)

        # Vmap over blocks
        model_params, _ = nk.jax.tree_ravel(vs.parameters.unfreeze())
        block_dp = update_blocks(model_params, vs.samples, block_idx, grad_block, ema_block)
        
        # Combine block updates
        dp = jnp.zeros_like(grad)
        dp = jax.tree_util.tree_map(lambda i, j: dp.at[i].set(j), block_idx, block_dp)

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