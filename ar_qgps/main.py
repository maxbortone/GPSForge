import jax
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
from ar_qgps import train
from VMCutils import MPIVars, add_file_logger


FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string('workdir', None,
                               'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the system, Ansatz and optimiser configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if MPIVars.rank == 0:
        add_file_logger(_WORKDIR.value, basename=FLAGS.config.trainer)

        logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
        logging.info('JAX local devices: %r', jax.local_devices())
        jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend)
        logging.info('Using JAX XLA backend %s', jax_xla_backend)
        logging.info('Config: %s', FLAGS.config)

    if FLAGS.config.trainer == 'vmc':
        train.vmc(FLAGS.config, _WORKDIR.value)
    elif FLAGS.config.trainer == 'ar_state_fitting':
        train.ar_state_fitting(FLAGS.config, _WORKDIR.value)
    else:
        raise app.UsageError(f'Unknown trainer: {FLAGS.config.trainer}')

if __name__ == '__main__':
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)