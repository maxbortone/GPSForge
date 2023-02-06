import os
import time
import ml_collections
import numpy as np
import netket as nk
import GPSKet as qk
from absl import logging
from optax.experimental import split_real_and_imaginary
from ar_qgps.datasets import get_dataset
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from VMCutils import MPIVars, write_config, Timer
from VMCutils import save_checkpoint, restore_checkpoint


def ar_state_fitting(config: ml_collections.ConfigDict, workdir: str):
    """Fits an autoregressive Ansatz to a target wavefunction."""

    # Load dataset
    dataset = get_dataset(config.system_name, config.dataset)

    # Setup system
    if config.dataset.basis != config.system.basis:
        config.system.basis = config.dataset.basis
    ha = get_system(config.system_name, config.system)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config.model_name, config.model, hi, g)

    # Sampler
    if config.system_name in ['Heisenberg1d', 'Heisenberg2d', 'J1J22d']:
        sa_dtype = np.int8
    elif config.system_name in ['Hchain', 'H2O']:
        sa_dtype = np.uint8
    sa = qk.sampler.ARDirectSampler(hi, **config.sampler, dtype=sa_dtype)

    # Variational state
    if config.variational_state_name == 'MCState':
        vs = nk.vqs.MCState(sa, ma, **config.variational_state)
    elif config.variational_state_name == 'ExactState':
        vs = nk.vqs.ExactState(hi, ma)
    elif config.variational_state_name == 'MCStateUniqeSamples':
        vs = qk.vqs.MCStateUniqeSamples(sa, ma, **config.variational_state)

    # Optimizer
    if 'Sgd' in config.optimizer_name:
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
    elif config.optimizer_name == 'Adam':
        op = nk.optimizer.Adam(learning_rate=config.optimizer.learning_rate, b1=config.optimizer.b1, b2=config.optimizer.b2)
    if config.model.dtype == 'complex':
        op = split_real_and_imaginary(op)

    # Restore checkpoint
    parameters = vs.parameters.unfreeze()
    opt_state = op.init(parameters)
    initial_step = 1
    opt_state, parameters, initial_step = restore_checkpoint(workdir, (opt_state, parameters, initial_step))
    vs.parameters = parameters
    if MPIVars.rank == 0:
        logging.info('Will start/continue training at initial_step=%d', initial_step)

    # Driver
    arsf = qk.driver.ARStateFitting(dataset, ha, op, variational_state=vs, mini_batch_size=config.mini_batch_size)

    # Setup logger and write config to file
    # TODO: replace JsonLog with HDF5Log
    logger = nk.logging.JsonLog(os.path.join(workdir, f"output_{initial_step}"), save_params=True, save_params_every=config.log_every, write_every=config.log_every)
    if MPIVars.rank == 0:
        write_config(workdir, config)
        logging.info(f"Saved config at {os.path.join(workdir, 'config.yaml')}")

    # Run training loop
    if MPIVars.rank == 0:
        logging.info('Starting training loop; initial compile can take a while...')
        timer = Timer(config.total_steps)
        t0 = time.time()
    total_steps = config.total_steps
    for step in range(initial_step, total_steps + 1):

        arsf.advance()
        logger(step, {"Loss": arsf._loss_stats}, arsf.state)

        if MPIVars.rank == 0 and step == initial_step:
            logging.info('First step took %.1f seconds.', time.time() - t0)
            t0 = time.time()

        if MPIVars.rank == 0:
            timer.update(step)

        # Report training metrics
        if MPIVars.rank == 0 and config.progress_every and step % config.progress_every == 0:
            grad, _ = nk.jax.tree_ravel(arsf._loss_grad)
            grad_norm = np.linalg.norm(grad)
            done = step / total_steps
            logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '  # pylint: disable=logging-format-interpolation
                         f'L: {arsf.loss}, '
                         f'||∇E||: {grad_norm:.4f}, '
                         f'{timer}')

        # Store checkpoint
        if MPIVars.rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == total_steps):
            checkpoint_path = save_checkpoint(workdir, (arsf._optimizer_state, arsf.state.parameters, step), step, overwrite=True)
            logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    return 