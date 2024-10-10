import os
import time
import ml_collections
import numpy as np
import netket as nk
import GPSKet as qk
from absl import logging
from optax.contrib import split_real_and_imaginary
from gps_forge.datasets import get_dataset
from gps_forge.systems import get_system
from gps_forge.models import get_model
from gps_forge.variational_states import get_variational_state
from VMCutils import Timer, CSVLogger
from netket.utils.mpi import rank as mpi_rank
from flax import serialization
from flax.training.checkpoints import save_checkpoint, restore_checkpoint


def serialize_ARStateFitting(driver: qk.driver.ARStateFitting):
    state_dict = {
        "variables": serialization.to_state_dict(driver.state.variables),
        "optimizer": serialization.to_state_dict(driver._optimizer_state),
        "step": driver._step_count
    }
    return state_dict

def deserialize_ARStateFitting(driver: nk.driver.VMC, state_dict: dict):
    import copy

    new_driver = copy.copy(driver)
    new_driver.state.variables = serialization.from_state_dict(driver.state.variables, state_dict["variables"])
    new_driver._optimizer_state = serialization.from_state_dict(driver._optimizer_state, state_dict["optimizer"])
    new_driver._step_count = serialization.from_state_dict(driver._step_count, state_dict["step"])
    return new_driver

serialization.register_serialization_state(
    qk.driver.ARStateFitting,
    serialize_ARStateFitting,
    deserialize_ARStateFitting
)

def ar_state_fitting(config: ml_collections.ConfigDict, workdir: str):
    """Fits an autoregressive Ansatz to a target wavefunction."""

    # Load dataset
    dataset = get_dataset(config.system_name, config.dataset)

    # Setup system
    if config.dataset.basis != config.system.basis:
        raise ValueError(f"The basis set of the system and the one used to generate the data don't match.")
    ha = get_system(config)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config, hi, g)

    # Sampler
    if config.system_name in ['Hchain', 'H2O']:
        sa_dtype = np.uint8
    sa = qk.sampler.ARDirectSampler(hi, **config.sampler, dtype=sa_dtype)

    # Variational state
    vs = get_variational_state(config, ma, hi, sa)

    # Optimizer
    if 'Sgd' in config.optimizer_name:
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
    elif config.optimizer_name == 'Adam':
        op = nk.optimizer.Adam(learning_rate=config.optimizer.learning_rate, b1=config.optimizer.b1, b2=config.optimizer.b2)
    if config.model.dtype == 'complex':
        op = split_real_and_imaginary(op)

    # Driver
    arsf = qk.driver.ARStateFitting(dataset, ha, op, variational_state=vs, mini_batch_size=config.mini_batch_size)

    # Restore checkpoint
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    arsf = restore_checkpoint(checkpoints_dir, arsf)
    initial_step = arsf.step_count + 1
    step = initial_step
    if mpi_rank == 0:
        logging.info('Will start/continue training at initial_step=%d', initial_step)

    # Logger
    if mpi_rank == 0:
        fieldnames = ["Loss", "Runtime"]
        logger = CSVLogger(os.path.join(workdir, "metrics.csv"), fieldnames)

    # Run training loop
    if mpi_rank == 0:
        logging.info('Starting training loop; initial compile can take a while...')
        timer = Timer(config.total_steps)
        t0 = time.time()
    total_steps = config.total_steps
    for step in range(initial_step, total_steps + 1):
        # Training step
        arsf.advance()

        # Report compilation time
        if mpi_rank == 0 and step == initial_step:
            logging.info(f"First step took {time.time() - t0:.1f} seconds.")
        
        # Update timer
        if mpi_rank == 0:
            timer.update(step)
        
        # Log data
        if mpi_rank == 0:
            logger(step, {"Loss": arsf._loss_stats, "Runtime": timer.runtime})

        # Report training metrics
        if mpi_rank == 0 and config.progress_every and step % config.progress_every == 0:
            grad, _ = nk.jax.tree_ravel(arsf._loss_grad)
            grad_norm = np.linalg.norm(grad)
            done = step / total_steps
            logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                         f"L: {arsf.loss}, "
                         f"||∇E||: {grad_norm:.4f}, "
                         f"{timer}")

        # Store checkpoint
        if mpi_rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == total_steps):
            checkpoint_path = save_checkpoint(workdir, arsf, step, keep_every_n_steps=config.checkpoint_every)
            logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    return 