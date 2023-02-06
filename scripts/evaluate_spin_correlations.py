
import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
import time
import jax
import numpy as np
import netket as nk
import GPSKet as qk
from absl import app
from absl import flags
from absl import logging
from ar_qgps.systems import get_system
from ar_qgps.models import get_model
from VMCutils import MPIVars, Timer, add_file_logger
from VMCutils import read_config, restore_checkpoint


FLAGS = flags.FLAGS

WORKDIR = flags.DEFINE_string('workdir', None, 'Directory to the model data.')
N_SAMPLES = flags.DEFINE_integer('n_samples', 1000, 'Number of samples used to evaluate the spin corelations')
CENTRAL_SITE = flags.DEFINE_integer('central_site', 0, 'Site for which the spin corelations are evaluated')

flags.mark_flag_as_required('workdir')

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Parse flags
    workdir = WORKDIR.value
    n_samples = N_SAMPLES.value
    central_site = CENTRAL_SITE.value

    if MPIVars.rank == 0:
        add_file_logger(workdir, basename="spin_correlations")

    # Get config
    config = read_config(workdir)
    Lx = config.system.Lx
    Ly = config.system.Ly
    if MPIVars.rank == 0:
        n_sites = Lx*Ly
        if central_site < 0 or central_site >= n_sites:
            raise ValueError(f"Central site must be a value between 0 and {n_sites-1}")
        logging.info('Config: \n%s', config)

    # Setup system and model
    ha = get_system(config.system_name, config.system)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None
    ma = get_model(config.model_name, config.model, hi, g)
    sa_cls = {
        'MetropolisLocal': nk.sampler.MetropolisLocal,
        'MetropolisExchange': nk.sampler.MetropolisExchange,
        'MetropolisHopping': qk.sampler.MetropolisHopping,
        'ARDirectSampler': qk.sampler.ARDirectSampler
    }[config.get('sampler_name', 'ARDirectSampler')]
    kwargs = config.to_dict()['sampler']
    if config.system_name in ['Hchain', 'H2O']:
        kwargs['dtype'] = np.uint8
    if config.sampler_name == 'MetropolisExchange':
        kwargs['graph'] = g
    sa = sa_cls(hi, **kwargs)
    if config.variational_state_name == 'MCState':
        vs = nk.vqs.MCState(sa, ma, **config.variational_state)
    else:
        raise ValueError("{config.variational_state_name} is not supported")
    params = vs.parameters.unfreeze()
    _, params, _ = restore_checkpoint(workdir, (None, params, None))
    vs.parameters = params
    if n_samples != config.variational_state.n_samples:
        vs.n_samples = n_samples

    # Setup spin correlation operator
    sz_sz = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    exchange = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    op = sz_sz - exchange if config.system.sign_rule else sz_sz + exchange

    # Compute spin correlations
    if MPIVars.rank == 0:
        logging.info('Starting loop; initial compile can take a while...')
        timer = Timer(config.total_steps)
        t0 = time.time()
    correlations = np.zeros((Lx, Ly))
    for step, site in enumerate(g.sites):
        # Compute spin-spin correlation
        corr_op = nk.operator.LocalOperator(hi, operators=op, acting_on=[central_site, site.id])
        corr = vs.expect(corr_op)

        # Report compilation time
        if MPIVars.rank == 0 and step == 0:
            logging.info('First step took %.1f seconds.', time.time() - t0)
        
        # Update timer
        if MPIVars.rank == 0:
            timer.update(step+1)

        # Log data
        x, y = site.position.astype(np.int32)
        correlations[x, y] = corr.mean.real
        
        # Report progress
        if MPIVars.rank == 0:
            logging.info(f"Step: {step+1}/{n_sites}, 〈S_{central_site}S_{site.id}〉 = {corr.mean.real}, {timer}")

    # Save result
    if MPIVars.rank == 0:
        np.savetxt(os.path.join(workdir, f"spin_correlations_{central_site}.txt"), correlations)
        logging.info(f"Saved spin correlation data at {workdir}")



if __name__ == '__main__':
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)