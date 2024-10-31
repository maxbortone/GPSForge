import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags, ConfigDict
from netket.utils.mpi import rank as mpi_rank
from netket.utils.mpi import n_nodes as mpi_n_nodes
from gps_forge.configs.common import resolve
from gps_forge.systems import get_system
from gps_forge.models import get_model
from gps_forge.samplers import get_sampler
from gps_forge.variational_states import get_variational_state
from VMCutils import add_file_logger, read_config
from flax.training.checkpoints import restore_checkpoint


FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string(
    'workdir',
    None,
    'Directory to the model data.',
    required=True
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the system, Ansatz and optimiser configuration.',
    lock_config=True
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Parse flags
    workdir = _WORKDIR.value
    config = _CONFIG.value
    filename = os.path.join(workdir, "config.yaml")
    if os.path.isfile(filename) and config is None:
        config = ConfigDict(read_config(workdir))
    config = resolve(config)

    if mpi_rank == 0:
        add_file_logger(workdir, basename="evaluate_qgt_rank")

    logging.info(f"JAX local devices: {jax.local_devices()}")
    if mpi_rank == 0:
        platform = jax.lib.xla_bridge.get_backend().platform
        if platform == "gpu":
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            logging.info(f"{result.stdout}")
        logging.info(f"JAX device count: {mpi_n_nodes}")
        jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend)
        logging.info(f"Using JAX XLA backend {jax_xla_backend}")
        logging.info(f"Config: \n{config}")

    # Setup system and model
    ha = get_system(config, workdir=workdir)
    hi = ha.hilbert
    ma = get_model(config, hi, hamiltonian=ha, workdir=workdir)
    sa = get_sampler(config, hi)
    vs = get_variational_state(config, ma, hilbert=hi, sampler=sa)

    # Load last checkpoint
    checkpoint = restore_checkpoint(workdir, None)
    params = jax.tree_util.tree_map(lambda x: jnp.array(x, x.dtype), checkpoint['1'])
    vs.parameters = params

    # Compute QGT rank
    qgt = nk.optimizer.qgt.QGTJacobianDense(vs, mode='real')
    S = qgt.to_dense()
    rank = jnp.linalg.matrix_rank(S)

    # Save result
    if mpi_rank == 0:
        np.save(os.path.join(workdir, "qgt_rank.npy"), np.array(rank))
        logging.info(f"Saved QGT rank at {workdir}")


if __name__ == '__main__':
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)