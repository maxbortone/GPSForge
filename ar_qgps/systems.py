import os
import numpy as np
import netket as nk
import GPSKet as qk
from ml_collections import ConfigDict
from netket.operator import AbstractOperator, Heisenberg
from GPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly
from GPSKet.operator.hamiltonian import FermiHubbardOnTheFly
from pyscf import scf, gto, ao2mo, lo
from pyscf.mcscf.casci import CASCI
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
    elif name in ['Hchain', 'Hsheet', 'H2O', 'Cr2', 'Cr']:
        if config.system.get('frozen_electrons', None) is not None:
            return get_frozen_core_molecular_system(config.system, workdir=workdir)
        else:
            return get_molecular_system(config.system, workdir=workdir)
    elif 'Hubbard' in name:
        return get_Hubbard_system(config.system)
    else:
        raise ValueError(f"Could not find system with name {name}")

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

        mf = scf.RHF(mol)
        mf.scf()
        norb = mf.mo_coeff.shape[1]
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
        h1_path = os.path.join(workdir, "h1.npy")
        h2_path = os.path.join(workdir, "h2.npy")
        if (os.path.exists(basis_path) and
                os.path.exists(h1_path) and
                os.path.exists(h2_path)):
            basis = np.load(basis_path)
            h1 = np.load(h1_path)
            h2 = np.load(h2_path)
        else:
            # Transform to a local orbital basis if wanted
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
                    localizer = lo.Boys(mol, mf.mo_coeff[:,:nelec//2])
                    loc_coeff_occ = localizer.kernel()
                    localizer = lo.Boys(mol, mf.mo_coeff[:, nelec//2:])
                    loc_coeff_vrt = localizer.kernel()
                    loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)
                basis = loc_coeff
            elif config.basis == 'canonical':
                basis = mf.mo_coeff
            else:
                raise ValueError("Unknown basis, please choose between: 'canonical', 'local-boys', 'local-pipek-mezey', 'local-edmiston-ruedenberg' and 'local-split'.")
            ovlp = mf.get_ovlp()
            # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
            assert(np.allclose(np.linalg.multi_dot((basis.T, ovlp, basis)), np.eye(norb)))
            # Find the hamiltonian the basis
            h1 = np.linalg.multi_dot((basis.T, mf.get_hcore(), basis))
            h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), norb)
            np.save(basis_path, basis)
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

def get_frozen_core_molecular_system(config : ConfigDict, workdir : str=None) -> AbInitioHamiltonianOnTheFly:
    """
    Return the Hamiltonian for a molecular system with a frozen core

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
        frozen_electrons = config.frozen_electrons
        if config.get('n_elec', None):
            n_elec = (config.n_elec[0]-frozen_electrons//2, config.n_elec[1]-frozen_electrons//2)
            spin = n_elec[0]-n_elec[1]
        else:
            spin = 0
        mol = gto.Mole()
        mol.build(
            atom=atom,
            basis=config.basis_set,
            symmetry=config.symmetry,
            unit=config.unit,
            spin=spin
        )
        nelec = mol.nelectron-frozen_electrons
        if config.get('n_elec', None) is None:
            n_elec = (nelec//2, nelec//2)
        mf = scf.RHF(mol)
        if config.get('sfx2c1e', None):
            mf = scf.sfx2c1e(mf)
        mf.scf()
        norb = mf.mo_coeff.shape[1]-frozen_electrons//2
        print(f"Number of active molecular orbitals: {norb}")
        print(f"Number of active α and β electrons: {n_elec}")
    else:
        norb = None
        n_elec = None
        nelec = None
    norb = MPIVars.comm.bcast(norb, root=0) # Number of active molecular orbitals
    n_elec = MPIVars.comm.bcast(n_elec, root=0) # Number of active α and β electrons
    nelec = MPIVars.comm.bcast(nelec, root=0) # Total number of active electrons

    hi = qk.hilbert.FermionicDiscreteHilbert(N=norb, n_elec=n_elec)

    # Get hamiltonian elements
    if MPIVars.rank == 0:
        # Load hamiltonian elements if they exist
        if workdir is None:
            workdir = os.getcwd()
        basis_path = os.path.join(workdir, "basis.npy")
        h1_path = os.path.join(workdir, "h1.npy")
        h2_path = os.path.join(workdir, "h2.npy")
        e_core_path = os.path.join(workdir, "e_core.npy")
        hf_orbitals_path = os.path.join(workdir, "hf_orbitals.npy")
        if (os.path.exists(basis_path) and
                os.path.exists(h1_path) and
                os.path.exists(h2_path) and
                os.path.exists(e_core_path) and
                os.path.exists(hf_orbitals_path)):
            basis = np.load(basis_path)
            h1 = np.load(h1_path)
            h2 = np.load(h2_path)
            e_core = np.load(e_core_path)
            hf_orbitals = np.load(hf_orbitals_path)
        else:
            # Compute molecular orbitals in active space
            casci = CASCI(mf, norb, nelec)
            mo_coeff = casci.mo_coeff[
                :, casci.ncore : casci.ncore + casci.ncas
            ]
            # Transform to a local orbital basis if wanted
            if 'local' in config.basis:
                if 'boys' in config.basis:
                    localizer = lo.Boys(mol, mo_coeff=mo_coeff)
                    localizer.init_guess = None
                    loc_coeff = localizer.kernel()
                basis = loc_coeff
            elif config.basis == 'canonical':
                basis = mo_coeff
            else:
                raise ValueError("Unknown basis, please choose between: 'canonical' and 'local-boys'.")
            ovlp = mf.get_ovlp()
            # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
            assert(np.allclose(np.linalg.multi_dot((basis.T, ovlp, basis)), np.eye(norb)))
            # Find the hamiltonian the basis
            h1, e_core = casci.get_h1cas()
            h2 = ao2mo.restore(1, casci.get_h2cas(), norb)
            canonical_to_local_trafo = basis.T.dot(ovlp.dot(mo_coeff))
            if 'local' in config.basis:
                h1 = np.linalg.multi_dot(
                    (canonical_to_local_trafo, h1, canonical_to_local_trafo.T)
                )
                h2 = np.einsum(
                    "ijkl,ai,bj,ck,dl->abcd",
                    h2,
                    canonical_to_local_trafo,
                    canonical_to_local_trafo,
                    canonical_to_local_trafo,
                    canonical_to_local_trafo,
                    optimize=True,
                )
            hf_orbitals = canonical_to_local_trafo[:, :nelec//2]
            np.save(basis_path, basis)
            np.save(h1_path, h1)
            np.save(h2_path, h2)
            np.save(e_core_path, e_core)
            np.save(hf_orbitals_path, hf_orbitals)
    else:
        h1 = None
        h2 = None
    h1 = MPIVars.comm.bcast(h1, root=0)
    h2 = MPIVars.comm.bcast(h2, root=0)

    # Setup Hamiltonian
    ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)
    return ha

def get_Hubbard_system(config: ConfigDict, return_graph: bool=False) -> FermiHubbardOnTheFly:
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
    if Ly > 1:
        if 'APBC' in config.pbc:
            for i, edge in enumerate(edges):
                if np.abs(edge[0]-edge[1]) // config.Ly == (config.Lx-1):
                    t[i] *= -1.0
    else:
        if config.pbc and Lx % 4 == 0:
            t[-1] *= -1.0
    ha = FermiHubbardOnTheFly(hi, edges, U=U, t=t)
    if return_graph:
        return ha, g
    else:
        return ha
