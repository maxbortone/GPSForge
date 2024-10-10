import os
import jax
import subprocess
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags, ConfigDict
from gps_forge import train
from gps_forge.configs.common import resolve
from netket.utils.mpi import rank as mpi_rank
from netket.utils.mpi import n_nodes as mpi_n_nodes
from VMCutils import add_file_logger, read_config, write_config


FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string(
    'workdir',
    None,
    'Directory to store logs and model data.',
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

    # Parse config
    workdir = _WORKDIR.value
    config = _CONFIG.value
    filename = os.path.join(workdir, "config.yaml")
    if os.path.isfile(filename) and config is None:
        config = ConfigDict(read_config(workdir))
    config = resolve(config)

    if mpi_rank == 0:
        add_file_logger(workdir, basename=config.trainer)

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
        write_config(workdir, config)

    if config.trainer == 'vmc':
        train.vmc(config, workdir)
    elif config.trainer == 'vmc_block':
        train.vmc_block(config, workdir)
    elif config.trainer == 'fssc':
        train.fssc(config, workdir)
    elif config.trainer == 'ar_state_fitting':
        train.ar_state_fitting(config, workdir)
    elif config.trainer == 'benchmark':
        train.benchmark(config, workdir)
    else:
        raise app.UsageError(f"Unknown trainer: {config.trainer}")

if __name__ == '__main__':
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)