import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
from absl import app
from absl import flags
from absl import logging
from pyscf import gto, scf, cc
from ml_collections import config_flags
from gps_forge.configs import systems
from gps_forge.configs.common import resolve
from gps_forge.models import setup_mol
from gps_forge.systems import get_Hubbard_system
from pyscf import scf, ao2mo, fci


FLAGS = flags.FLAGS

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
    hamiltonian = get_Hubbard_system(config.system)
    mol, h1, h2, norb, n_elec, nelec = setup_mol(config, hamiltonian)

    # Compute FCI energy
    energy_mo, _ = fci.direct_spin0.FCI().kernel(h1, h2, norb, nelec)
    energy_nuc = mol.energy_nuc()
    exact_energy = energy_mo + energy_nuc

    logging.info(f"Nuclear energy {energy_nuc}")
    logging.info(f"FCI total energy {exact_energy}")

if __name__ == '__main__':
    app.run(main)