import os
import numpy as np
import netket as nk
import GPSKet as qk
from ml_collections import ConfigDict
from netket.operator import AbstractOperator, Heisenberg
from GPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly
from GPSKet.operator.hamiltonian import FermiHubbardOnTheFly
from pyscf import scf, gto, ao2mo, lo
from VMCutils import MPIVars


def get_system(config : ConfigDict, workdir : str=None) -> AbstractOperator:
    """
    Return the Hamiltonian for a system

    Args:
        config : experiment configuration file
        workdir : working directory (optional)

    Returns:
        Hamiltonian for the system
    """
    name = config.system_name
    if 'Heisenberg' in name or 'J1J2' in name:
        return get_Heisenberg_system(config.system)
    elif name in ['Hchain', 'Hsheet', 'H2O']:
        return get_molecular_system(config.system, workdir=workdir)
    elif 'Hubbard' in name:
        return get_Hubbard_system(config.system)

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
    Ly = config.get('Ly', None)
    J1 = config.J1
    J2 = config.get('J2', 0.0)

    # Setup Hamiltonian
    if J2 != 0.0:
        sign_rule = (config.sign_rule, False)
    else:
        sign_rule = config.sign_rule
    ha = qk.operator.hamiltonian.get_J1_J2_Hamiltonian(Lx, Ly=Ly, J1=J1, J2=J2, total_sz=config.total_sz, sign_rule=sign_rule, pbc=config.pbc, on_the_fly_en=True)
    return ha

def get_molecular_system(config : ConfigDict, workdir : str=None) -> AbInitioHamiltonianOnTheFly:
    """
    Return the Hamiltonian for a molecular system

    Args:
        config : system configuration dictionary
        workdir : working directory

    Returns:
        Hamiltonian for the molecular system
    """
    # Setup Hilbert space
    if config.get('atom', None):
        atom = config.atom
    else:
        atom = config.molecule
    if MPIVars.rank == 0:
        mol = gto.Mole()
        mol.build(
            atom=atom,
            basis=config.basis_set,
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
        # Transform to a local orbital basis if wanted
        if workdir is None:
            workdir = os.getcwd()
        basis_path = os.path.join(workdir, "basis.npy")
        if os.path.exists(basis_path):
            loc_coeff = np.load(basis_path)
        else:
            if 'local' in config.basis:
                loc_coeff = lo.orth_ao(mol, 'lowdin')
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
            elif config.basis == 'canonical':
                loc_coeff = myhf.mo_coeff
            else:
                raise ValueError("Unknown basis, please choose between: 'canonical', 'local-boys', 'local-pipek-mezey', 'local-edmiston-ruedenberg' and 'local-split'.")
            np.save(basis_path, loc_coeff)
        h1_path = os.path.join(workdir, "h1.npy")
        h2_path = os.path.join(workdir, "h2.npy")
        if os.path.exists(h1_path) and os.path.exists(h2_path):
            h1 = np.load(h1_path)
            h2 = np.load(h2_path)
        else:
            ovlp = myhf.get_ovlp()
            # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
            assert(np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)),np.eye(norb)))
            # Find the hamiltonian the basis
            h1 = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff))
            h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)
            np.save(h1_path, h1)
            np.save(h2_path, h2)
    else:
        h1 = None
        h2 = None
    h1 = MPIVars.comm.bcast(h1, root=0)
    h2 = MPIVars.comm.bcast(h2, root=0)

    # Setup Hamiltonian
    ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)
    return ha

def get_Hubbard_system(config: ConfigDict) -> FermiHubbardOnTheFly:
    """
    Return the Hamiltonian for Hubbard system at half-filling

    Args:
        config : system configuration dictionary

    Returns:
        Hamiltonian for the Hubbard system at half-filling
    """
    # Setup graph and Hilbert space
    Lx = config.Lx
    Ly = config.get('Ly', 1)
    t = config.t
    U = config.U
    n_elec = config.get('n_elec', None)
    if Ly > 1:
        # config.pbc:
        # - 'PBC-PBC' = periodic boundary conditions in both dimensions
        # - 'PBC-APBC' = periodic boundary conditions in one dimension, and anti-periodic in the other
        #   - implemented by changing the sign on the hopping terms in one direction
        # - 'PBC-OBC' = periodic boundary conditions in one dimension, and open in the other
        if config.pbc in ['PBC', 'PBC-PBC', 'PBC-APBC']:
            pbc = [True, True]
        elif 'OBC' in config.pbc:
            pbc = [True, False] 
        g = nk.graph.Grid([Lx, Ly], pbc=pbc)
    else:
        # config.pbc = True/False
        g = nk.graph.Chain(Lx, pbc=config.pbc)
    if n_elec is None:
        n_elec = (g.n_nodes//2, g.n_nodes//2)
    hi = qk.hilbert.FermionicDiscreteHilbert(g.n_nodes, n_elec=n_elec)

    # Setup Hamiltonian
    edges = np.array(g.edges())
    t = np.ones(edges.shape[0])*config.t
    if Ly > 1 and 'APBC' in config.pbc:
        for i, edge in enumerate(edges):
            if np.abs(edge[0]-edge[1]) // config.Ly == (config.Lx-1):
                t[i] *= -1.0
    else:
        if config.pbc and Lx % 4 == 0:
            t[-1] *= -1.0
    ha = FermiHubbardOnTheFly(hi, edges, U=U, t=t)
    return ha
