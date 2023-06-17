import os
import time
import ml_collections
import numpy as np
import netket as nk
import GPSKet as qk
import jax.numpy as jnp
from absl import logging
from typing import Union
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from ar_qgps.samplers import get_sampler
from ar_qgps.variational_states import get_variational_state
from ar_qgps.optimizers import get_optimizer
from VMCutils import MPIVars, Timer, CSVLogger
from VMCutils import restore_best_params, save_best_params
from flax import serialization
from flax.training.checkpoints import save_checkpoint, restore_checkpoint


def serialize_VMC(driver: nk.driver.VMC):
    # TODO: serialize sampler state
    state_dict = {
        "variables": serialization.to_state_dict(driver.state.variables),
        "optimizer": serialization.to_state_dict(driver._optimizer_state),
        "preconditioner": None,
        "step": driver._step_count
    }
    # TODO: improve this by including the predconditioner state in the optimizer state
    if type(driver.preconditioner).__name__ == "SRRMSProp":
        state_dict["preconditioner"] = serialization.to_state_dict(driver.preconditioner._ema)
    return state_dict

def deserialize_VMC(driver: nk.driver.VMC, state_dict: dict):
    import copy

    new_driver = copy.copy(driver)
    new_driver.state.variables = serialization.from_state_dict(driver.state.variables, state_dict["variables"])
    new_driver._optimizer_state = serialization.from_state_dict(driver._optimizer_state, state_dict["optimizer"])
    new_driver._step_count = serialization.from_state_dict(driver._step_count, state_dict["step"])
    if type(driver.preconditioner).__name__ == "SRRMSProp":
        new_driver.preconditioner._ema = serialization.from_state_dict(driver.preconditioner._ema, state_dict["preconditioner"])
    return new_driver

serialization.register_serialization_state(
    nk.driver.VMC,
    serialize_VMC,
    deserialize_VMC
)

serialization.register_serialization_state(
    qk.driver.minSRVMC,
    serialize_VMC,
    deserialize_VMC
)

def vmc(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using VMC."""

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

    # Restore checkpoint
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    vmc = restore_checkpoint(checkpoints_dir, vmc)
    initial_step = vmc.step_count+1
    step = initial_step
    if MPIVars.rank == 0:
        logging.info(f"Will start/continue training at initial_step={initial_step}")

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
    total_steps = config.total_steps
    for step in range(initial_step, total_steps + 1):
        # Training step
        vmc.advance()

        # Report compilation time
        if MPIVars.rank == 0 and step == initial_step:
            logging.info(f"First step took {time.time() - t0:.1f} seconds.")

        # Update timer
        if MPIVars.rank == 0:
            timer.update(step)

        # Log data
        if MPIVars.rank == 0:
            logger(step, {**vmc.energy.to_dict(), "Runtime": timer.runtime})

        # Save best energy params
        if MPIVars.rank == 0 and vmc.energy.mean.real < best_energy and vmc.energy.variance < best_variance:
            best_energy = vmc.energy.mean.real
            best_variance = vmc.energy.variance
            save_best_params(workdir, {"Energy": best_energy, "Variance": best_variance, "Parameters": vmc.state.parameters})
            logging.info(f"Stored best parameters at step {step} with energy {vmc.energy}")

        # Report training metrics
        if MPIVars.rank == 0 and config.progress_every and step % config.progress_every == 0:
            grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
            grad_norm = np.linalg.norm(grad)
            done = step / total_steps
            logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                         f"E: {vmc.energy}, "
                         f"||∇E||: {grad_norm:.4f}, "
                         f"{timer}")

        # Store checkpoint
        if MPIVars.rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == total_steps):
            checkpoint_path = save_checkpoint(checkpoints_dir, vmc, step, keep_every_n_steps=config.checkpoint_every)
            logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    return 