import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
import pyscf
import numpy as np
import pandas as pd
from ml_collections import ConfigDict
from gps_forge.configs.systems import get_config
from gps_forge.configs.common import resolve
from gps_forge.systems import build_molecule
try:
    from vayesta.ewf import EWF
except:
    raise ImportError("Activate an environment with Vayesta installed")


# Loop over dissociation distances
energies = []
converged = []
distances = np.linspace(1.0, 3.0, 21)
for d in distances:
    # Setup config
    config = get_config("Hsheet")
    config.system.basis = "local-boys"
    config.system.basis_set = "sto-6g"
    config.system.n_atoms = 36
    config.system.unit = "angstrom"
    config.system.distance = d
    config.vayesta = ConfigDict()
    config.vayesta.n_fragments = 36
    config.vayesta.bath = ConfigDict()
    config.vayesta.bath.bathtype = "dmet"
    config = resolve(config)
    print(config)

    # Build molecule
    mol = build_molecule(config.system)

    # Run Hartree-Fock on the full system
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    converged.append(mf.converged)

    # Run FCI on the full system for comparison
    if config.system.n_atoms <= 16:
        fci = pyscf.fci.FCI(mf)
        fci.kernel()

    # Run single-shot EWF embedding with FCI solver
    bath_options = config.vayesta.bath.to_dict()
    emb = EWF(mf, solver="FCI", bath_options=bath_options, solver_options=dict(conv_tol=1.e-14))
    # Set up fragments
    with emb.iaopao_fragmentation() as f:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
    emb.kernel()
    e_dmet_fci = emb.get_dmet_energy(with_exxdiv=False)
    energies.append(e_dmet_fci)

    print(f"RHF energy per atom: {mf.e_tot/config.system.n_atoms}")
    if config.system.n_atoms <= 16:
        print(f"FCI energy per atom: {fci.e_tot/config.system.n_atoms}")
    print(f"Vayesta DMET energy per atom: {e_dmet_fci/config.system.n_atoms}")

# Save energies
df = pd.DataFrame(data={'basis': [config.system.basis_set]*len(distances), 'D': distances, 'E': energies, 'converged': converged})
df.to_csv('result_Vayesta_DMET_FCI_H6x6.csv', index=False)
