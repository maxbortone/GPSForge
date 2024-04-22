import os
import time
import warnings
import jax
import flax
import ml_collections
import numpy as np
import netket as nk
import jax.numpy as jnp
from absl import logging
from functools import partial
from typing import Optional
from netket.utils.types import PyTree
from netket.jax import HashablePartial, vjp, tree_cast
from netket.stats import Stats
from netket.driver.vmc_common import info
from netket.vqs.mc.mc_state.state import check_chunk_size, _is_power_of_two
from GPSKet.operator.hamiltonian.ab_initio import local_en_on_the_fly
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from ar_qgps.samplers import get_sampler
from ar_qgps.variational_states import get_variational_state
from ar_qgps.optimizers import get_optimizer
from VMCutils import MPIVars, Timer, CSVLogger
from VMCutils import restore_best_params, save_best_params
from flax import core as fcore
from flax import serialization
from flax.training.checkpoints import save_checkpoint, restore_checkpoint


class FSSC(nk.driver.AbstractVariationalDriver):
    def __init__(self, hamiltonian, variational_state, optimizer, chunk_size: Optional[int]=None):
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")
        self._ham = hamiltonian
        self._get_conn_fun = lambda x: self._ham.get_conn_padded(x)[0]
        self._norb = self._ham.hilbert.size
        self._n_elec = self._ham.hilbert._n_elec
        self._nelec = jnp.sum(jnp.array(self._n_elec))
        self._n_unique = self.state.n_samples
        self.chunk_size = chunk_size
        self._dp: PyTree = None
        
        # Apply, log_probability and local_value_kernel functions
        self._apply_fun = HashablePartial(
            lambda model, vars, x, **kwargs: model.apply(vars, x, **kwargs),
            self.state.model
        )
        def log_proba_fun(variables, x, **kwargs):
            log_proba = 2 * self._apply_fun(variables, x, **kwargs)[0].real
            log_proba = log_proba - jax.scipy.special.logsumexp(log_proba)
            return log_proba
        try:
            use_fast_update = self.state.model.apply_fast_update
        except:
            use_fast_update = False
        if use_fast_update:
            self._apply_fun_kwargs = fcore.freeze(dict(mutable="intermediates_cache", cache_intermediates=True))
            self._log_proba_fun = HashablePartial(
                log_proba_fun,
                mutable="intermediates_cache",
                cache_intermediates=True
            )
        else:
            self._apply_fun_kwargs = fcore.freeze({})
            self._log_proba_fun = log_proba_fun
        self._local_value_args = (jnp.array(self._ham.t_mat), jnp.array(self._ham.eri_mat))
        self._local_value_kernel = HashablePartial(
            local_en_on_the_fly,
            self._n_elec,
            use_fast_update=use_fast_update,
            chunk_size=self.chunk_size,
        )

        # Init core space
        self.core_space = self._get_initial_core_space()

    @property
    def n_unique(self) -> int:
        """The total number of unique samples in the core space."""
        return self._n_unique

    @property
    def chunk_size(self) -> int:
        """
        Suggested *maximum size* of the chunks used in forward and backward evaluations
        of the Neural Network model.

        If your inputs are smaller than the chunk size this setting is ignored.

        This can be used to lower the memory required to run a computation with a very
        high number of samples or on a very large lattice. Notice that inputs and
        outputs must still fit in memory, but the intermediate computations will now
        require less memory.

        This option comes at an increased computational cost. While this cost should
        be negligible for large-enough chunk sizes, don't use it unless you are memory
        bound!

        This option is an hint: only some operations support chunking. If you perform
        an operation that is not implemented with chunking support, it will fall back
        to no chunking. To check if this happened, set the environment variable
        `NETKET_DEBUG=1`.
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: Optional[int]):
        # disable chunks if it is None
        if chunk_size is None:
            self._chunk_size = None
            return

        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer. ")

        if not _is_power_of_two(chunk_size):
            warnings.warn(
                "For performance reasons, we suggest to use a power-of-two chunk size.",
                stacklevel=2,
            )

        check_chunk_size(self.n_unique, chunk_size)

        self._chunk_size = chunk_size

    def _get_initial_core_space(self):
        # Get connected space of HF configuration
        hf_conf = jnp.zeros(self._norb, jnp.int32)
        hf_conf = hf_conf.at[:self._nelec//2].set(3)
        hf_conf = jnp.expand_dims(hf_conf, axis=0)
        x_conn = self._get_conn_fun(hf_conf)
        x_conn = jnp.squeeze(x_conn, axis=0)

        # Keep only unique configurations
        core_space = jnp.unique(jnp.concatenate((hf_conf, x_conn), axis=0), axis=0, size=self.n_unique, fill_value=-1)

        # Check how many configurations are still needed
        n_samples = jnp.argwhere(core_space[:,0] == -1, size=1)[0][0].item()
        if n_samples > 0:
            n_left = self.n_unique - n_samples 
        else:
            n_left = 0
            n_samples = self.n_unique
        logging.info("Building initial core space:")
        logging.info(f"{n_samples}/{self.n_unique}")

        # Loop until n_unique samples have been found
        while n_left > 0:
            # Find connected space of samples in core space
            x_conn = self._get_conn_fun(core_space[:n_samples])
            x_conn = jnp.reshape(x_conn, (x_conn.shape[0] * x_conn.shape[1], -1))

            # Keep only unique configurations
            core_space = jnp.unique(jnp.concatenate((core_space[:n_samples], x_conn), axis=0), axis=0, size=self.n_unique, fill_value=-1)

            # Check how many configurations are still needed
            n_samples = jnp.argwhere(core_space[:,0] == -1, size=1)[0][0].item()
            if n_samples > 0:
                n_left = self.n_unique - n_samples 
            else:
                n_left = 0
                n_samples = self.n_unique
            logging.info(f"{n_samples}/{self.n_unique}")
        logging.info("Completed!")
        return core_space

    # TODO: make this function jittable by refactoring AbInitioHamiltonianOnTheFly into a JAX-based operator
    def _update_core_space(self, params: PyTree, model_state: PyTree):
        # Get connected space of current core space
        start = time.time()
        conn_space = self._get_conn_fun(self.core_space)
        end = time.time()
        print(f"`get_conn_fun` took {end-start:.2f} seconds to find {np.prod(conn_space.shape[:-1])} connected configurations")
        conn_space = jnp.reshape(conn_space, (conn_space.shape[0] * conn_space.shape[1], -1))

        # Keep only unique configurations
        union_size = self.n_unique + conn_space.shape[0] * conn_space.shape[1]
        union_space = jnp.unique(jnp.concatenate((self.core_space, conn_space), axis=0), axis=0, size=union_size, fill_value=-1)

        # Evaluate amplitudes of configurations in union space
        idx_fill = jnp.argwhere(union_space[:,0] == -1, size=1)[0][0].item()
        if idx_fill > 0:
            union_space = union_space[:idx_fill]
        start = time.time()
        log_amps, _ = self._apply_fun({'params': params, **model_state}, union_space, **self._apply_fun_kwargs)
        end = time.time()
        print(f"`apply_fun` took {end-start:.2f} seconds to compute {np.prod(union_space.shape[:-1])} log-amplitudes")

        # Select n_unique largest amplitudes
        idx = jnp.argsort(2 * log_amps.real)[-self.n_unique:]

        # Update core space by selecting the corresponding configurations
        core_space = union_space[idx]
        return core_space
    
    def _forward_and_backward(self):
        # Get model state and params
        model_state, params = fcore.pop(self.state.variables, 'params')

        # Update core space
        if self.step_count > 0:
            self.core_space = self._update_core_space(
                params, model_state   
            )

        # Estimate energy and gradient
        en, grad = energy_and_grad(
            self._local_value_kernel,
            self._apply_fun,
            self._apply_fun_kwargs,
            self._log_proba_fun,
            params,
            model_state,
            self.core_space,
            self._local_value_args
        )
        self._loss_stats = Stats(mean=en, error_of_mean=0.0, variance=0.0)
        self._dp = tree_cast(grad, params)
        return self._dp
    
    @property
    def energy(self) -> Stats:
        """
        Return the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats
    
    def __repr__(self):
        return (
            "FSSC("
            + f"\n  step_count = {self.step_count},"
            + f"\n  n_unique = {self.n_unique})"
        )
    
    def info(self, depth=0):
        lines = [
            f"{name}: {info(obj, depth=depth + 1)}"
            for name, obj in [
                ("Hamiltonian    ", self._ham),
                ("Optimizer      ", self._optimizer),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self), *lines])
    
@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def energy_and_grad(
    local_value_kernel,
    apply_fun,
    apply_fun_kwargs,
    log_proba_fun,
    params,
    model_state,
    core_space,
    local_value_args
):  
    # Estimate energy
    O_loc = local_value_kernel(
        apply_fun,
        {"params": params, **model_state},
        core_space,
        local_value_args
    )
    probas = jnp.exp(log_proba_fun({"params": params, **model_state}, core_space))
    Ō = jnp.sum(probas * O_loc)

    # Estimate gradient
    _, vjp_fun, _ = vjp(
        lambda w, x: apply_fun({"params": w, **model_state}, x, **apply_fun_kwargs),
        params,
        core_space,
        conjugate=True,
        has_aux=True
    )
    O_loc_centered = probas * (O_loc - Ō)
    Ō_grad = vjp_fun(jnp.conjugate(O_loc_centered))[0]
    Ō_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        params,
    )
    return Ō, Ō_grad


flax.config.update('flax_use_orbax_checkpointing', False)

def serialize_FSSC(driver: FSSC):
    state_dict = {
        "core_space": serialization.to_state_dict(driver.core_space),
        "variables": serialization.to_state_dict(driver.state.variables),
        "optimizer": serialization.to_state_dict(driver._optimizer_state),
        "step": driver._step_count
    }
    return state_dict

def deserialize_FSSC(driver: FSSC, state_dict: dict):
    import copy

    new_driver = copy.copy(driver)
    new_driver.core_space = serialization.from_state_dict(driver.core_space, state_dict["core_space"])
    new_driver.state.variables = serialization.from_state_dict(driver.state.variables, state_dict["variables"])
    new_driver._optimizer_state = serialization.from_state_dict(driver._optimizer_state, state_dict["optimizer"])
    new_driver._step_count = serialization.from_state_dict(driver._step_count, state_dict["step"])
    return new_driver

serialization.register_serialization_state(
    FSSC,
    serialize_FSSC,
    deserialize_FSSC
)

def fssc(config: ml_collections.ConfigDict, workdir: str):
    """Trains an Ansatz on a system using FSSC."""

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
    op, _ = get_optimizer(config)

    # Driver
    fssc = FSSC(
        ha,
        vs,
        op,
        chunk_size=config.variational_state.chunk_size
    )

    # Restore checkpoint
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    fssc = restore_checkpoint(checkpoints_dir, fssc)
    initial_step = fssc.step_count+1
    step = initial_step
    if MPIVars.rank == 0:
        logging.info(f"Will start/continue training at initial_step={initial_step}")

    # Logger
    if MPIVars.rank == 0:
        fieldnames = list(nk.stats.Stats().to_dict().keys())+["Runtime"]
        logger = CSVLogger(os.path.join(workdir, "metrics.csv"), fieldnames)

    # Run training loop
    if initial_step < config.total_steps:
        if MPIVars.rank == 0:
            logging.info(f"Model has {vs.n_parameters} parameters")
            logging.info('Starting training loop; initial compile can take a while...')
            timer = Timer(config.total_steps)
            t0 = time.time()
            best_params = restore_best_params(workdir)
            best_energy = best_params["Energy"] if best_params else np.inf
        for step in range(initial_step, config.total_steps + 1):
            # Training step
            fssc.advance()

            # Report compilation time
            if MPIVars.rank == 0 and step == initial_step:
                logging.info(f"First step took {time.time() - t0:.1f} seconds.")

            # Update timer
            if MPIVars.rank == 0:
                timer.update(step)

            # Log data
            if MPIVars.rank == 0:
                logger(step, {**fssc.energy.to_dict(), "Runtime": timer.runtime})

            # Save best energy params
            if MPIVars.rank == 0 and fssc.energy.mean.real < best_energy:
                best_energy = fssc.energy.mean.real
                save_best_params(workdir, {"Energy": best_energy, "Parameters": fssc.state.parameters})
                logging.info(f"Stored best parameters at step {step} with energy {fssc.energy}")

            # Report training metrics
            if MPIVars.rank == 0 and config.progress_every and step % config.progress_every == 0:
                if hasattr(fssc, "_loss_grad"):
                    grad, _ = nk.jax.tree_ravel(fssc._loss_grad)
                    grad_norm = np.linalg.norm(grad)
                else:
                    grad_norm = np.nan
                done = step / config.total_steps
                logging.info(f"Step: {step}/{config.total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                            f"E: {fssc.energy}, "
                            f"||∇E||: {grad_norm:.4f}, "
                            f"{timer}")

            # Store checkpoint
            if MPIVars.rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == config.total_steps):
                # TODO: migrate to new orbax API (see: https://flax.readthedocs.io/en/latest/guides/use_checkpointing.htm)
                checkpoint_path = save_checkpoint(checkpoints_dir, fssc, step, keep_every_n_steps=config.checkpoint_every)
                logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    return 