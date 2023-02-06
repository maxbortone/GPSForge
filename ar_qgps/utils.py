import os
import pathlib
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from typing import Union, Tuple
from ml_collections import ConfigDict
from netket.operator import AbstractOperator
from GPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly
from pyscf import gto, fci, scf


def get_Heisenberg_exact_energy(config: ConfigDict, hamiltonian : AbstractOperator=None) -> Union[float, None]:
    """
    Returns the exact energy for a Heisenberg system defined by `config` if present in ./data folder,
    else attempts to diagonalize the Hamiltonian.
    If this also fails, returns None
    """
    exact_energy = None
    base_path = pathlib.Path(__file__).parent.parent.resolve()
    Lx = config.Lx
    Ly = config.get('Ly', 1)
    J1 = config.J1
    J2 = config.get('J2', 0.0)
    if Ly == 1:
        path = os.path.join(base_path, 'data/result_DMRG_Heisenberg_1D.csv')
        df = pd.read_csv(path, dtype={'L': np.int16, 'J': np.float32, 'E': np.float32})
        row = df.query('L == @Lx and J == @J1')
        exact_energy = row['E'].values[0]
    else:
        path = os.path.join(base_path, 'data/result_ED_J1J2_2D.csv')
        df = pd.read_csv(path, skiprows=0, header=1, dtype={'Lx': np.int16, 'Ly': np.int16, 'J1': np.float32, 'J2': np.float32, 'E/N': np.float32, 'E': np.float32})
        row = df.query('Lx == @Lx and Ly == @Ly and J1 == @J1 and J2 == @ J2')
        exact_energy = row['E'].values[0]
    if exact_energy is None and hamiltonian is not None:
        exact_energy = eigsh(hamiltonian.to_sparse(), k=1, which='SA', return_eigenvectors=False)[0]
    return exact_energy

def get_molecular_exact_energy(config: ConfigDict, hamiltonian: AbInitioHamiltonianOnTheFly) -> Tuple[float, float]:
    """
    Returns the exact energy for a molecular system defined by `config` and `hamiltonian`, computed with a FCI solver
    """
    n_electrons = np.sum(hamiltonian.hilbert._n_elec)
    n_orbitals = hamiltonian.hilbert.size
    mol = gto.Mole()
    if isinstance(config.atom, str) and hasattr(config, 'distance') and hasattr(config, 'n_atoms'):
        atom = [(config.atom, (x*config.distance, 0., 0.)) for x in range(config.n_atoms)]
    else:
        atom = config.atom
    mol.build(
        atom = atom,
        basis = config.basis_set,
        symmetry=config.symmetry,
        unit=config.unit
    )
    energy_mo, _ = fci.direct_spin1.FCI().kernel(hamiltonian.h_mat, hamiltonian.eri_mat, n_orbitals, n_electrons)
    energy_nuc = mol.energy_nuc()
    exact_energy = energy_mo + energy_nuc
    return exact_energy, energy_nuc

def get_molecular_hf_energy(config: ConfigDict) -> Tuple[float, float]:
    """
    Returns the Hartree-Fock energy for a molecular system defined by `config` and `hamiltonian`
    """
    mol = gto.Mole()
    if isinstance(config.atom, str) and hasattr(config, 'distance') and hasattr(config, 'n_atoms'):
        atom = [(config.atom, (x*config.distance, 0., 0.)) for x in range(config.n_atoms)]
    else:
        atom = config.atom
    mol.build(
        atom = atom,
        basis = config.basis_set,
        symmetry=config.symmetry,
        unit=config.unit
    )
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    return hf_energy
