import os
import time
import jax
import flax
import ml_collections
import numpy as np
import netket as nk
from netket.experimental.driver import VMC_SRt
from GPSKet.driver import VMC_SRRMSProp
import GPSKet as qk
import jax.numpy as jnp
from absl import logging
from gps_forge.systems import get_system
from gps_forge.models import get_model
from gps_forge.samplers import get_sampler
from gps_forge.variational_states import get_variational_state
from gps_forge.optimizers import get_optimizer
from VMCutils import Timer, CSVLogger
from VMCutils import restore_best_params, save_best_params
from netket.utils.mpi import rank as mpi_rank
from flax import serialization
from flax.training.checkpoints import save_checkpoint, restore_checkpoint


flax.config.update('flax_use_orbax_checkpointing', False)

def serialize_VMC(driver: nk.driver.VMC):
    # TODO: serialize sampler state
    state_dict = {
        "variables": serialization.to_state_dict(driver.state.variables),
        "optimizer": serialization.to_state_dict(driver._optimizer_state),
        "preconditioner": None,
        "step": driver._step_count
    }
    # TODO: improve this by including the predconditioner state in the optimizer state
    if hasattr(driver, "preconditioner") and type(driver.preconditioner).__name__ == "SRRMSProp":
        state_dict["preconditioner"] = serialization.to_state_dict(driver.preconditioner._ema)
    return state_dict

def deserialize_VMC(driver: nk.driver.VMC, state_dict: dict):
    import copy

    new_driver = copy.copy(driver)
    new_driver.state.variables = serialization.from_state_dict(driver.state.variables, state_dict["variables"])
    new_driver._optimizer_state = serialization.from_state_dict(driver._optimizer_state, state_dict["optimizer"])
    new_driver._step_count = serialization.from_state_dict(driver._step_count, state_dict["step"])
    if hasattr(driver, "preconditioner") and type(driver.preconditioner).__name__ == "SRRMSProp":
        new_driver.preconditioner._ema = serialization.from_state_dict(driver.preconditioner._ema, state_dict["preconditioner"])
    return new_driver

def serialize_VMC_SRRMSProp(driver: VMC_SRRMSProp):
    # TODO: serialize sampler state
    state_dict = {
        "variables": serialization.to_state_dict(driver.state.variables),
        "optimizer": serialization.to_state_dict(driver._optimizer_state),
        "ema": serialization.to_state_dict(driver.preconditioner._ema),
        "step": driver._step_count
    }
    return state_dict

def deserialize_VMC_SRRMSProp(driver: VMC_SRRMSProp, state_dict: dict):
    import copy

    new_driver = copy.copy(driver)
    new_driver.state.variables = serialization.from_state_dict(driver.state.variables, state_dict["variables"])
    new_driver._optimizer_state = serialization.from_state_dict(driver._optimizer_state, state_dict["optimizer"])
    new_driver._step_count = serialization.from_state_dict(driver._step_count, state_dict["step"])
    new_driver._ema = serialization.from_state_dict(driver._ema, state_dict["ema"])
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

serialization.register_serialization_state(
    VMC_SRt,
    serialize_VMC,
    deserialize_VMC
)

serialization.register_serialization_state(
    VMC_SRRMSProp,
    serialize_VMC,
    deserialize_VMC
)

def vmc(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using VMC."""

    # Setup system
    ha = get_system(config, workdir=workdir)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Ansatz model
    ma = get_model(config, hi, g, ha, workdir)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Variational state
    vs = get_variational_state(config, ma, hi, sa)

    # Optimizer
    op, sr = get_optimizer(config, vs)

    # Driver
    if config.optimizer_name == 'minSR':
        solver = lambda A, b: jnp.linalg.lstsq(A, b, rcond=config.optimizer.rcond)[0]
        vmc = qk.driver.minSRVMC(ha, op, variational_state=vs, mode=config.optimizer.mode, minSR_solver=solver)
    elif config.optimizer_name == 'kernelSR':
        vmc = VMC_SRt(ha, op, variational_state=vs, jacobian_mode=config.optimizer.mode, diag_shift=config.optimizer.diag_shift)
    elif config.optimizer_name == 'SRRMSProp':
        vmc = VMC_SRRMSProp(ha, op, variational_state=vs, diag_shift=config.optimizer.diag_shift, decay=config.optimizer.decay, eps=config.optimizer.eps, jacobian_mode=config.optimizer.mode)
    else:
        vmc = nk.driver.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Restore checkpoint
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    vmc = restore_checkpoint(checkpoints_dir, vmc)
    initial_step = vmc.step_count+1
    step = initial_step
    if mpi_rank == 0:
        logging.info(f"Will start/continue training at initial_step={initial_step}")

    # Logger
    if mpi_rank == 0:
        fieldnames = list(nk.stats.Stats().to_dict().keys())+["Runtime"]
        logger = CSVLogger(os.path.join(workdir, "metrics.csv"), fieldnames)

    # Run training loop
    if initial_step < config.total_steps:
        if mpi_rank == 0:
            logging.info(f"Model has {vs.n_parameters} parameters")
            logging.info('Starting training loop; initial compile can take a while...')
            timer = Timer(config.total_steps)
            t0 = time.time()
            best_params = restore_best_params(workdir)
            best_energy = best_params["Energy"] if best_params else np.inf
            best_variance = best_params["Variance"] if best_params else np.inf
        for step in range(initial_step, config.total_steps + 1):
            # Training step
            vmc.advance()
            if hasattr(vmc.state.sampler_state, 'acceptance'):
                acceptance = vmc.state.sampler_state.acceptance
            else:
                acceptance = 1.0

            # Report compilation time
            if mpi_rank == 0 and step == initial_step:
                logging.info(f"First step took {time.time() - t0:.1f} seconds.")

            # Update timer
            if mpi_rank == 0:
                timer.update(step)

            # Log data
            if mpi_rank == 0:
                logger(step, {**vmc.energy.to_dict(), "Runtime": timer.runtime})

            # Save best energy params
            if mpi_rank == 0 and vmc.energy.mean.real < best_energy and vmc.energy.variance < best_variance:
                best_energy = vmc.energy.mean.real
                best_variance = vmc.energy.variance
                save_best_params(workdir, {"Energy": best_energy, "Variance": best_variance, "Parameters": vmc.state.parameters})
                logging.info(f"Stored best parameters at step {step} with energy {vmc.energy}")

            # Report training metrics
            if mpi_rank == 0 and config.progress_every and step % config.progress_every == 0:
                if hasattr(vmc, "_loss_grad"):
                    grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
                    grad_norm = np.linalg.norm(grad)
                else:
                    grad_norm = np.nan
                done = step / config.total_steps
                logging.info(f"Step: {step}/{config.total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                            f"E: {vmc.energy}, "
                            f"||∇E||: {grad_norm:.4f}, "
                            f"acceptance: {acceptance*100:.2f}%, "
                            f"{timer}")

            # Store checkpoint
            if mpi_rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == config.total_steps):
                # TODO: migrate to new orbax API (see: https://flax.readthedocs.io/en/latest/guides/use_checkpointing.htm)
                checkpoint_path = save_checkpoint(checkpoints_dir, vmc, step, keep_every_n_steps=config.checkpoint_every)
                logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    # Descent finishing
    if config.get('descent_finishing', None) and initial_step < config.total_steps + config.descent_finishing.total_steps:
        # Switch to first SGD optimizer
        op = nk.optimizer.Sgd(learning_rate=config.descent_finishing.learning_rate)
        vmc.optimizer = op

        # Run training loop
        if mpi_rank == 0:
            logging.info('Starting descent finishing loop...')
            timer = Timer(config.descent_finishing.total_steps)
            t0 = time.time()
            best_params = restore_best_params(workdir)
            best_energy = best_params["Energy"] if best_params else np.inf
            best_variance = best_params["Variance"] if best_params else np.inf
        total_steps = config.total_steps + config.descent_finishing.total_steps
        for step in range(step+1, total_steps + 1):
            # Training step
            vmc.advance()
            if hasattr(vmc.state.sampler_state, 'acceptance'):
                acceptance = vmc.state.sampler_state.acceptance
            else:
                acceptance = 1.0

            # Report compilation time
            if mpi_rank == 0 and step == 1:
                logging.info(f"First step took {time.time() - t0:.1f} seconds.")

            # Update timer
            if mpi_rank == 0:
                timer.update(step)

            # Log data
            if mpi_rank == 0:
                logger(step, {**vmc.energy.to_dict(), "Runtime": timer.runtime})

            # Save best energy params
            if mpi_rank == 0 and vmc.energy.mean.real < best_energy and vmc.energy.variance < best_variance:
                best_energy = vmc.energy.mean.real
                best_variance = vmc.energy.variance
                save_best_params(workdir, {"Energy": best_energy, "Variance": best_variance, "Parameters": vmc.state.parameters})
                logging.info(f"Stored best parameters at step {step} with energy {vmc.energy}")

            # Report training metrics
            if mpi_rank == 0 and config.progress_every and step % config.progress_every == 0:
                if hasattr(vmc, "_loss_grad"):
                    grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
                    grad_norm = np.linalg.norm(grad)
                else:
                    grad_norm = np.nan
                done = step / total_steps
                logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                            f"E: {vmc.energy}, "
                            f"||∇E||: {grad_norm:.4f}, "
                            f"acceptance: {acceptance*100:.2f}%, "
                            f"{timer}")

            # Store checkpoint
            if mpi_rank == 0:
                # TODO: migrate to new orbax API (see: https://flax.readthedocs.io/en/latest/guides/use_checkpointing.htm)
                checkpoint_path = save_checkpoint(checkpoints_dir, vmc, step)
                logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    # Evaluate model
    if config.get('evaluate', None):
        # Restore model
        best = restore_best_params(workdir)
        if best is not None:
            best_params = best["Parameters"]
            best_params = jax.tree_map(lambda x: jnp.array(x, x.dtype), best_params)
            vs.parameters = best_params

        # Update variational state settings
        vs.n_samples = config.evaluate.n_samples
        vs.chunk_size = config.evaluate.chunk_size

        # Logger
        if mpi_rank == 0:
            fieldnames = list(nk.stats.Stats().to_dict().keys())+["n_samples", "Runtime"]
            logger = CSVLogger(os.path.join(workdir, "evals.csv"), fieldnames)

        # Run evaluation loop
        if mpi_rank == 0:
            logging.info('Starting evaluation loop...')
            timer = Timer(config.evaluate.total_steps)
            t0 = time.time()
        total_steps = config.evaluate.total_steps
        for step in range(1, total_steps + 1):
            # Evaluation step
            vs.reset()
            energy = vs.expect(ha)

            # Report compilation time
            if mpi_rank == 0 and step == 1:
                logging.info(f"First step took {time.time() - t0:.1f} seconds.")

            # Update timer
            if mpi_rank == 0:
                timer.update(step)

            # Log data
            if mpi_rank == 0:
                logger(step, {**energy.to_dict(), "n_samples": config.evaluate.n_samples, "Runtime": timer.runtime})

            # Report evaluation metrics
            if mpi_rank == 0:
                done = step / total_steps
                logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                            f"E: {energy}, "
                            f"{timer}")

    return 