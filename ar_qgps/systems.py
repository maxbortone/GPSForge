import numpy as np
import netket as nk
import qGPSKet as qk
from ml_collections import ConfigDict
from netket.operator import AbstractOperator, Heisenberg
from qGPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly
from pyscf import scf, gto, ao2mo, lo
from VMCutils import MPIVars


def get_system(name : str, config : ConfigDict) -> AbstractOperator:
    """
    Return the Hamiltonian for a system

    Args:
        name : name of the system
        config : system configuration dictionary

    Returns:
        Hamiltonian for the system
    """
    if 'Heisenberg' in name or 'J1J2' in name:
        return get_Heisenberg_system(config)
    elif name == 'Hchain' or name == 'H2O':
        return get_molecular_system(config)

def get_Heisenberg_system(config : ConfigDict) -> Heisenberg:
    """
    Return the Hamiltonian for Heisenberg system

    Args:
        config : system configuration dictionary

    Returns:
        Hamiltonian for the Heisenberg system
    """
    # Setup Hilbert space
    Lx = config.Lx
    Ly = config.get('Ly', 1)
    J1 = config.J1
    J2 = config.get('J2', 0.0)
    nb_order = 2 if J2 != 0.0 else 1
    extent = [Lx, Ly] if Ly > 1 else [Lx]
    g = nk.graph.Grid(extent, max_neighbor_order=nb_order, pbc=config.pbc)
    hi = nk.hilbert.Spin(0.5, total_sz=config.total_sz, N=g.n_nodes)

    # Setup Hamiltonian
    J = [J1/4, J2/4] if J2 != 0.0 else J1/4
    ha = nk.operator.Heisenberg(hi, g, J=J, sign_rule=config.sign_rule)
    return ha

def get_molecular_system(config : ConfigDict) -> AbInitioHamiltonianOnTheFly:
    """
    Return the Hamiltonian for a molecular system

    Args:
        config : system configuration dictionary

    Returns:
        Hamiltonian for the molecular system
    """
    # Setup Hilbert space
    if MPIVars.rank == 0:
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
        nelec = mol.nelectron
        print('Number of electrons: ', nelec)

        myhf = scf.RHF(mol)
        myhf.scf()
        norb = myhf.mo_coeff.shape[1]
        print('Number of molecular orbitals: ', norb)
    else:
        norb = None
        nelec = None
    norb = MPIVars.comm.bcast(norb, root=0)
    nelec = MPIVars.comm.bcast(nelec, root=0)

    hi = qk.hilbert.FermionicDiscreteHilbert(N=norb, n_elec=(nelec//2,nelec//2))

    # Get hamiltonian elements
    if MPIVars.rank == 0:
        # 1-electron 'core' hamiltonian terms, transformed into MO basis
        h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

        # Get 2-electron electron repulsion integrals, transformed into MO basis
        eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)

        # Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
        # Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
        h2 = ao2mo.restore(1, eri, norb)

        # Transform to a local orbital basis if wanted
        if 'local' in config.basis:
            loc_coeff = lo.orth_ao(mol, 'meta_lowdin')
            if 'boys' in config.basis:
                localizer = lo.Boys(mol, mo_coeff=loc_coeff)
                localizer.init_guess = None
                loc_coeff = localizer.kernel()
            elif 'pipek-mezey' in config.basis:
                loc_coeff = lo.PipekMezey(mol, mo_coeff=loc_coeff).kernel()
            elif 'edmiston-ruedenberg' in config.basis:
                loc_coeff = lo.EdmistonRuedenberg(mol, mo_coeff=loc_coeff).kernel()
            elif 'split' in config.basis:
                localizer = lo.Boys(mol, myhf.mo_coeff[:,:nelec//2])
                loc_coeff_occ = localizer.kernel()
                localizer = lo.Boys(mol, myhf.mo_coeff[:, nelec//2:])
                loc_coeff_vrt = localizer.kernel()
                loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)
            ovlp = myhf.get_ovlp()
            # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
            assert(np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)),np.eye(norb)))
            # Find the hamiltonian in the local basis
            hij_local = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
            hijkl_local = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
            h1 = hij_local
            h2 = hijkl_local
    else:
        h1 = None
        h2 = None
    h1 = MPIVars.comm.bcast(h1, root=0)
    h2 = MPIVars.comm.bcast(h2, root=0)

    # Setup Hamiltonian
    ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)
    return ha
