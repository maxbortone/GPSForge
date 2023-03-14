import os
import time
import jax
import ml_collections
import numpy as np
import netket as nk
import GPSKet as qk
import jax.numpy as jnp
from absl import logging
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from ar_qgps.samplers import get_sampler
from ar_qgps.variational_states import get_variational_state
from ar_qgps.optimizers import get_optimizer
from VMCutils import MPIVars, write_config, Timer
from VMCutils import save_checkpoint, restore_checkpoint, save_best_params


def vmc(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using VMC."""

    # Setup system
    ha = get_system(config)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config, hi, g)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Variational state
    vs = get_variational_state(config, ma, hi, sa)

    # Optimizer
    op, sr = get_optimizer(config, vs)

    # Restore checkpoint
    parameters = vs.parameters.unfreeze()
    opt_state = op.init(parameters)
    initial_step = 1
    opt_state, parameters, initial_step = restore_checkpoint(workdir, (opt_state, parameters, initial_step))
    vs.parameters = parameters
    if MPIVars.rank == 0:
        logging.info('Will start/continue training at initial_step=%d', initial_step)

    # Driver
    if config.optimizer_name == 'minSR':
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        solver = lambda x: jnp.linalg.pinv(x, rcond=config.optimizer.rcond, hermitian=True)
        vmc = qk.driver.minSRVMC(ha, op, variational_state=vs, mode=config.optimizer.mode, minSR_solver=solver)
    else:
        vmc = nk.driver.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Setup logger and write config to file
    # TODO: replace JsonLog with HDF5Log
    logger = nk.logging.JsonLog(os.path.join(workdir, f"output_{initial_step}"), save_params=False, write_every=config.log_every)
    if MPIVars.rank == 0:
        write_config(workdir, config)
        logging.info(f"Saved config at {os.path.join(workdir, 'config.yaml')}")

    # Run training loop
    if MPIVars.rank == 0:
        logging.info('Starting training loop; initial compile can take a while...')
        timer = Timer(config.total_steps)
        t0 = time.time()
        best_energy = np.inf
        best_variance = np.inf
    total_steps = config.total_steps
    for step in range(initial_step, total_steps + 1):
        # Training step
        vmc.advance()

        # Report compilation time
        if MPIVars.rank == 0 and step == initial_step:
            logging.info('First step took %.1f seconds.', time.time() - t0)

        # Update timer
        if MPIVars.rank == 0:
            timer.update(step)
            wallclock = timer.elapsed_time
        else:
            wallclock = None
        wallclock = MPIVars.comm.bcast(wallclock, root=0)

        # Log data
        logger(step, {"Energy": vmc._loss_stats, "Wallclock": wallclock}, vmc.state)

        # Save best energy params
        if MPIVars.rank == 0 and vmc.energy.mean.real < best_energy and vmc.energy.variance < best_variance:
            best_energy = vmc.energy.mean.real
            best_variance = vmc.energy.variance
            save_best_params(workdir, vmc.state.parameters)
            logging.info(f"Stored best parameters at step {step} with energy {vmc.energy}")

        # Report training metrics
        if MPIVars.rank == 0 and config.progress_every and step % config.progress_every == 0:
            grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
            grad_norm = np.linalg.norm(grad)
            done = step / total_steps
            logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '  # pylint: disable=logging-format-interpolation
                         f'E: {vmc.energy}, '
                         f'||∇E||: {grad_norm:.4f}, '
                         f'{timer}')

        # Store checkpoint
        if MPIVars.rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == total_steps):
            checkpoint_path = save_checkpoint(workdir, (vmc._optimizer_state, vmc.state.parameters, step), step, overwrite=True)
            logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    return 