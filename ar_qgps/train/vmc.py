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
from VMCutils import MPIVars, write_config, Timer
from VMCutils import save_checkpoint, restore_checkpoint


def vmc(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using VMC."""

    # Setup system
    ha = get_system(config.system_name, config.system)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config.model_name, config.model, hi, g)

    # Sampler
    if hasattr(config.model, 'normalize'):
        if not config.model.normalize and config.sampler_name == 'ARDirectSampler':
            raise ValueError("The ARDirectSampler can only be used with normalized autoregressive models")
    sa_cls = {
        'MetropolisLocal': nk.sampler.MetropolisLocal,
        'MetropolisExchange': nk.sampler.MetropolisExchange,
        'MetropolisHopping': qk.sampler.MetropolisHopping,
        'ARDirectSampler': qk.sampler.ARDirectSampler
    }[config.get('sampler_name', 'ARDirectSampler')]
    kwargs = config.to_dict()['sampler']
    if config.system_name in ['Hchain', 'H2O', 'Hubbard1d']:
        kwargs['dtype'] = np.uint8
    if config.sampler_name == 'MetropolisExchange':
        kwargs['graph'] = g
    sa = sa_cls(hi, **kwargs)

    # Variational state
    if config.variational_state_name == 'MCState':
        vs = nk.vqs.MCState(sa, ma, **config.variational_state)
    elif config.variational_state_name == 'ExactState':
        vs = nk.vqs.ExactState(hi, ma)
    elif config.variational_state_name == 'MCStateUniqeSamples':
        vs = qk.vqs.MCStateUniqueSamples(sa, ma, **config.variational_state)
    elif config.variational_state_name == 'MCStateStratifiedSampling':
        sa = qk.sampler.MetropolisHopping(hi, n_sweeps=config.variational_state.n_sweeps, n_chains_per_rank=1)
        if MPIVars.rank == 0:
            from ar_qgps.datasets import get_dataset

            dataset = get_dataset(config.system_name, config.variational_state.dataset)
            det_set_size = config.variational_state.deterministic_set_size
            norb = dataset[0].shape[1]
            det_set = np.zeros((det_set_size, norb), dtype=np.uint8)
            det_inds = np.argsort(np.abs(dataset[1]))[:-det_set_size-1:-1]
            np.copyto(det_set, dataset[0][det_inds,:])
            hilbert_size = dataset[0].shape[0]
        else:
            hilbert_size = None
            det_set = None
        hilbert_size = MPIVars.comm.bcast(hilbert_size, root=0)
        det_set = MPIVars.comm.bcast(det_set, root=0)
        vs = qk.vqs.MCStateStratifiedSampling(
            det_set, hilbert_size, sa, ma,
            number_random_samples=config.variational_state.n_random_samples,
            n_samples=config.variational_state.n_samples,
            chunk_size=config.variational_state.chunk_size,
            n_discard_per_chain=config.variational_state.n_discard_per_chain,
            renormalize=config.variational_state.renormalize,
            rand_norm=config.variational_state.rand_norm)

    # Optimizer
    if config.optimizer_name == 'Sgd' or config.optimizer_name == 'minSR':
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        sr = None
    elif config.optimizer_name == 'Adam':
        op = nk.optimizer.Adam(learning_rate=config.optimizer.learning_rate, b1=config.optimizer.b1, b2=config.optimizer.b2)
        sr = None
    elif config.optimizer_name == 'SRDense':
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        qgt = nk.optimizer.qgt.QGTJacobianDense(mode=config.optimizer.mode, diag_shift=config.optimizer.diag_shift, diag_scale=config.optimizer.diag_scale)
        sr = qk.optimizer.SRDense(qgt)
    elif config.optimizer_name == 'SRRMSProp':
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        qgt = nk.optimizer.qgt.QGTJacobianDense(mode=config.optimizer.mode)
        sr = qk.optimizer.SRRMSProp(
            vs.parameters,
            qgt,
            diag_shift=config.optimizer.diag_shift,
            decay=config.optimizer.decay,
            eps=config.optimizer.eps
        )

    # Restore checkpoint
    parameters = vs.parameters.unfreeze()
    opt_state = op.init(parameters)
    initial_step = 1
    opt_state, parameters, initial_step = restore_checkpoint(workdir, (opt_state, parameters, initial_step))
    vs.parameters = parameters
    if MPIVars.rank == 0:
        logging.info('Will start/continue training at initial_step=%d', initial_step)

    # Driver
    if 'minSR' in config.optimizer_name:
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