import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from absl import app
from absl import flags
from absl import logging
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from ar_qgps.samplers import get_sampler
from ar_qgps.variational_states import get_variational_state
from GPSKet.operator.hamiltonian.ab_initio import local_en_on_the_fly
from VMCutils import MPIVars, read_config, restore_best_params


FLAGS = flags.FLAGS

WORKDIR = flags.DEFINE_string('workdir', None, 'Directory to the model data.')

flags.mark_flag_as_required('workdir')

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Parse flags
    workdir = WORKDIR.value

    # Get config
    config = read_config(workdir)

    # Setup system and model
    ha = get_system(config, workdir=workdir)
    hi = ha.hilbert
    n_elec = hi._n_elec
    norb = ha.hilbert.size
    ma = get_model(config, hi, hamiltonian=ha, workdir=workdir)
    sa = get_sampler(config, hi)
    vs = get_variational_state(config, ma, hilbert=hi, sampler=sa)

    # Load best parameters
    best_params = restore_best_params(workdir)['Parameters']
    best_params = jax.tree_map(lambda x: jnp.array(x, x.dtype), best_params)
    vs.parameters = best_params

    # Sample state
    n_samples = config.evaluate.n_samples
    vs.chunk_size = config.evaluate.chunk_size
    vs.reset()
    samples = vs.sample(n_samples=n_samples)
    samples = samples.reshape((-1, norb))

    # Compute RDMs
    try:
        use_fast_update = vs.model.apply_fast_update
    except:
        use_fast_update = False
    _, local_rdm1, local_rdm2 = local_en_on_the_fly(
        n_elec,
        vs._apply_fun,
        vs.variables,
        samples,
        (jnp.array(ha.t_mat), jnp.array(ha.eri_mat)),
        use_fast_update=use_fast_update,
        chunk_size=config.evaluate.chunk_size,
        return_local_RDMs=True
    )
    rdm1, _ = nk.utils.mpi.mpi_sum_jax(jnp.sum(local_rdm1, axis=0))
    rdm1 = np.array(rdm1).real / n_samples
    rdm2, _ = nk.utils.mpi.mpi_sum_jax(jnp.sum(local_rdm2, axis=0))
    rdm2 = np.array(rdm2).real / n_samples

    # Re-order into (p^+ r^+ s q) form
    rdm2 *= 2
    for k in range(norb):
        rdm2[:, k, k, :] -= rdm1.T

    # Save result
    if MPIVars.rank == 0:
        np.save(os.path.join(workdir, "rdm_1.npy"), rdm1)
        np.save(os.path.join(workdir, "rdm_2.npy"), rdm2)
        logging.info(f"Saved RDMs at {workdir}")


if __name__ == '__main__':
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)