import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
from absl import app
from absl import flags
from absl import logging
from pyscf import gto, scf, cc, fci
from pyscf.gto.basis import BasisNotFoundError
from ml_collections import config_flags
from ar_qgps.configs.common import resolve
from ar_qgps.systems import build_molecule, get_molecular_system


FLAGS = flags.FLAGS

method = flags.DEFINE_string('method', 'CCSD', 'QC method used to compute the energy. Choose between: HF, CCSD, CCSD(T) and FCI.')
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
    mol = build_molecule(config.system)
    logging.info(f"Nuclear energy {mol.energy_nuc()}")
    if method.value != "FCI":
        if restricted.value:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        if config.system.get('sfx2c1e', None):
            mf = mf.sfx2c1e()
        mf.run()
        logging.info(f"HF total energy {mf.e_tot}")
        if "CCSD" in method.value:
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
    else:
        ha = get_molecular_system(config.system)
        n_electrons = sum(ha.hilbert._n_elec)
        n_orbitals = ha.hilbert.size
        energy_mo, _ = fci.direct_spin1.FCI().kernel(ha.h_mat, ha.eri_mat, n_orbitals, n_electrons)
        energy_nuc = mol.energy_nuc()
        exact_energy = energy_mo + energy_nuc
        logging.info(f"FCI total energy {exact_energy}")
        

if __name__ == '__main__':
    app.run(main)