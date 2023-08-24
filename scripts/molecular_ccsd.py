import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
from absl import app
from absl import flags
from absl import logging
from pyscf import gto, scf, cc
from ml_collections import config_flags
from ar_qgps.configs.common import resolve


FLAGS = flags.FLAGS

method = flags.DEFINE_string('method', 'CCSD', 'CC method used to compute the energy. Choose between: CCSD and CCSD(T)')
restricted = flags.DEFINE_bool('restricted', True, 'Flag to choose between a calculation with restricted or unrestricted spin orbitals')

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
    if config.system.get('n_elec', None):
        n_elec = config.system.get('n_elec')
        spin = n_elec[0]-n_elec[1]
    else:
        spin = 0
    mol = gto.M(
        atom=config.system.molecule,
        basis=config.system.basis_set,
        symmetry=config.system.symmetry,
        unit=config.system.unit,
        spin=spin
    )
    logging.info(f"Nuclear energy {mol.energy_nuc()}")
    if restricted.value:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    if config.system.get('sfx2c1e', None):
        mf = mf.sfx2c1e()
    mf.run()
    frozen = config.system.get('frozen_electrons', 0)//2
    if restricted.value:
        mycc = cc.RCCSD(mf, frozen=frozen)
    else:
        mycc = cc.UCCSD(mf, frozen=frozen)
    mycc.run()
    logging.info(f"CCSD total energy {mycc.e_tot}")
    if method.value == "CCSD(T)":
        et = mycc.ccsd_t()
        logging.info(f"CCSD(T) total energy {mycc.e_tot + et}")

if __name__ == '__main__':
    app.run(main)