import os
import time
import ml_collections
import numpy as np
import netket as nk
import qGPSKet as qk
from absl import logging
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from VMCutils import MPIVars, write_config, Timer
from VMCutils import save_checkpoint, restore_checkpoint


def train(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using VMC."""

    # Setup system
    ha = get_system(config.system_name, config.system)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config.model_name, config.model, hi, g)

    # Sampler
    sa_cls = {
        'MetropolisLocal': nk.sampler.MetropolisLocal,
        'MetropolisExchange': nk.sampler.MetropolisExchange,
        'MetropolisHopping': qk.sampler.MetropolisHopping,
        'ARDirectSampler': qk.sampler.ARDirectSampler
    }[config.get('sampler_name', 'ARDirectSampler')]
    if config.system_name in ['Heisenberg1d', 'Heisenberg2d', 'J1J22d']:
        sa_dtype = np.int8
    elif config.system_name in ['Hchain', 'H2O']:
        sa_dtype = np.uint8
    sa = sa_cls(hi, **config.sampler, dtype=sa_dtype)

    # Variational state
    if config.variational_state_name == 'MCState':
        vs = nk.vqs.MCState(sa, ma, **config.variational_state)
    elif config.variational_state_name == 'ExactState':
        vs = nk.vqs.ExactState(hi, ma)

    # Optimizer
    sr = None
    if 'Sgd' in config.optimizer_name:
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
    elif config.optimizer_name == 'Adam':
        op = nk.optimizer.Adam(learning_rate=config.optimizer.learning_rate, b1=config.optimizer.b1, b2=config.optimizer.b2)
    if 'SR' in config.optimizer_name:
        if 'Dense' in config.optimizer_name:
            qgt = nk.optimizer.qgt.QGTJacobianDense(mode=config.optimizer.mode)
        else:
            qgt = nk.optimizer.qgt.QGTAuto()
        sr = nk.optimizer.SR(qgt=qgt, diag_shift=config.optimizer.diag_shift, iterative=config.optimizer.iterative)

    # Restore checkpoint
    parameters = vs.parameters.unfreeze()
    opt_state = op.init(parameters)
    initial_step = 1
    opt_state, parameters, initial_step = restore_checkpoint(workdir, (opt_state, parameters, initial_step))
    vs.parameters = parameters
    if MPIVars.rank == 0:
        logging.info('Will start/continue training at initial_step=%d', initial_step)

    # Variational Monte Carlo driver
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
    total_steps = config.total_steps
    for step in range(initial_step, total_steps + 1):

        vmc.advance()
        logger(step, {"Energy": vmc._loss_stats}, vmc.state)

        if MPIVars.rank == 0 and step == initial_step:
            logging.info('First step took %.1f seconds.', time.time() - t0)
            t0 = time.time()

        if MPIVars.rank == 0:
            timer.update(step)

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