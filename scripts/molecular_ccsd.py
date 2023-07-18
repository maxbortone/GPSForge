import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
from absl import app
from absl import flags
from absl import logging
from pyscf import gto, scf, cc
from ml_collections import config_flags
from ar_qgps.configs import systems
from ar_qgps.configs.common import resolve


FLAGS = flags.FLAGS

method = flags.DEFINE_string('method', 'ccsd', 'CC method used to compute the energy')

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the system configuration.',
    lock_config=True
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Get config
    config = _CONFIG.value
    config = resolve(config)
    logging.info('Config: %s', config)

    # Setup molecular system
    mol = gto.M(
        atom=config.system.molecule,
        basis=config.system.basis_set,
        symmetry=config.system.symmetry,
        unit=config.system.unit
    )
    logging.info(f"Nuclear energy {mol.energy_nuc()}")
    mf = scf.HF(mol).run()
    mycc = cc.CCSD(mf).run()
    logging.info(f"CCSD total energy {mycc.e_tot}")
    if method.value == "CCSD(T)":
        et = mycc.ccsd_t()
        logging.info(f"CCSD(T) total energy {mycc.e_tot + et}")

if __name__ == '__main__':
    app.run(main)