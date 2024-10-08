import os
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import GPSKet as qk
from scipy.linalg import circulant
from functools import partial
from flax import linen as nn
from netket.hilbert import HomogeneousHilbert
from netket.graph import AbstractGraph
from netket.utils import HashableArray
from netket.utils.types import Array
from ml_collections import ConfigDict
from pyscf import gto, scf, ao2mo
from GPSKet.operator.hamiltonian import AbInitioHamiltonianOnTheFly, FermiHubbardOnTheFly
from netket.utils.mpi import rank as mpi_rank
from netket.utils.mpi import mpi_bcast
from plum import dispatch
from typing import Union, Tuple, Callable, Optional


Hamiltonian = Union[AbInitioHamiltonianOnTheFly, FermiHubbardOnTheFly]

_MODELS = {
    'qGPS': qk.models.qGPS,
    'PlaquetteqGPS': qk.models.qGPS,
    'SegGPS': qk.models.SegGPS,
    'ARqGPS': qk.models.ARqGPS,
    'ARqGPSFull': qk.models.ARqGPSFull,
    'ARPlaquetteqGPS': qk.models.ARPlaquetteqGPS,
    'PixelCNN': qk.models.PixelCNN,
    'BackflowCPD': qk.models.Backflow,
    'CPDBackflow': qk.models.CPDBackflow,
    'SlaterqGPS': qk.models.SlaterqGPS,
}

def get_model(config : ConfigDict, hilbert : HomogeneousHilbert, graph : Optional[AbstractGraph]=None, hamiltonian : Hamiltonian = None, workdir: str=None) -> nn.Module:
    """
    Return the model for a wavefunction Ansatz

    Args:
        config : experiment configuration file
        hilbert : Hilbert space on which the model should act
        graph : graph associated with the Hilbert space (optional)
        hamiltonian : Hamiltonian of the system (optional)
        workdir : working directory (optional)

    Returns:
        the model for the wavefunction Ansatz
    """
    name = config.model_name
    try:
        ma_cls = _MODELS[name]
    except KeyError:
        raise ValueError(f"Model {name} is not a valid class or is not supported yet.")
    if config.model.dtype == 'real':
        dtype = jnp.float64
    elif config.model.dtype == 'complex':
        dtype = jnp.complex128
    if name != 'SegGPS' and name != 'SlaterqGPS' and 'GPS'  in name:
        if isinstance(config.model.M, tuple):
            assert len(config.model.M) == hilbert.size
            M = HashableArray(np.array(config.M))
        else:
            M = int(config.model.M)
        init_fn = qk.nn.initializers.normal(sigma=config.model.sigma, dtype=dtype)
        if graph:
            groups = config.model.symmetries.split(',')
            translations = 'translations' in groups or groups[0] == 'all' or groups[0] == 'automorphisms'
            point_symmetries = 'point-symmetries' in groups or groups[0] == 'all' or groups[0] == 'automorphisms'
            spin_flip = 'spin-flip' in groups or groups[0] == 'all' or groups[0] == 'automorphisms'
            symmetries_fn, inv_symmetries_fn = get_symmetry_transformation_spin(name, translations, point_symmetries, spin_flip, graph)
        else:
            symmetries_fn, inv_symmetries_fn = qk.models.no_syms()
        out_trafo = get_out_transformation(name, config.model.apply_exp)
        if 'AR' in name:
            if isinstance(hilbert, nk.hilbert.Spin):
                count_spins_fn = count_spins
                renormalize_log_psi_fn = renormalize_log_psi
            elif isinstance(hilbert, qk.hilbert.FermionicDiscreteHilbert):
                count_spins_fn = count_spins_fermionic
                renormalize_log_psi_fn = renormalize_log_psi_fermionic
            args = [hilbert, M]
            if 'Plaquette' in name:
                args.extend(get_plaquettes_and_masks(hilbert, graph))
            if 'Full' in name:
                apply_symmetries = (symmetries_fn, inv_symmetries_fn)
            else:
                apply_symmetries = symmetries_fn
            ma = ma_cls(
                    *args,
                    dtype=dtype,
                    init_fun=init_fn,
                    normalize=config.model.normalize,
                    apply_symmetries=apply_symmetries,
                    count_spins=count_spins_fn,
                    renormalize_log_psi=renormalize_log_psi_fn,
                    out_transformation=out_trafo)
        else:
            # Implement PlaquetteqGPS as a qGPS with kernel symmetrization over lattice translations
            # FIXME: this doesn't support projective symmetrization
            if 'Plaquette' in name:
                args = [hilbert, M]
                if config.system_name == 'Hubbard1d':
                    graph = nk.graph.Chain(config.system.Lx, pbc=config.system.pbc)
                symmetries_fn, inv_symmetries_fn = get_symmetry_transformation_spin(name, True, False, False, graph)
            else:
                args = [hilbert, hilbert.size*M]
            ma = ma_cls(
                *args,
                dtype=dtype,
                init_fun=init_fn,
                syms=(symmetries_fn, inv_symmetries_fn),
                out_transformation=out_trafo,
                apply_fast_update=True)
    elif name == 'SegGPS':
        M = int(config.model.M)
        init_fun = qk.nn.initializers.normal(config.model.sigma, dtype=dtype)
        ma = ma_cls(
            hilbert,
            M,
            dtype=dtype,
            init_fun=init_fun
        )
    elif 'PixelCNN' in name:
        if config.system.get('Ly', None) is None:
            raise ValueError("PixelCNN Ansatz is only implemented for 2D systems.")
        if isinstance(hilbert, nk.hilbert.Spin):
            def total_sz(inputs: Array) -> Array:
                n_spins_up = jnp.cumsum(inputs == -1., axis=-1)
                n_spins_dn = jnp.cumsum(inputs == 1., axis=-1)
                n_spins = jnp.stack([n_spins_up, n_spins_dn], axis=-1)
                n_spins = jnp.concatenate([jnp.zeros((inputs.shape[0], 1, 2)), n_spins[:, :-1, :]], axis=1)
                return n_spins

            def get_zero_total_sz_constraint(hilbert : HomogeneousHilbert):
                def constraint_fn(gauge: Array) -> Array: 
                    return gauge >= hilbert.size // 2
                return constraint_fn
            gauge_fn = total_sz
            constraint_fn = get_zero_total_sz_constraint(hilbert)
        elif isinstance(hilbert, qk.hilbert.FermionicDiscreteHilbert):
            raise ValueError("PixelCNN Ansatz is not implemented for fermionic systems.")
        ma = ma_cls(
            hilbert,
            param_dtype=dtype,
            kernel_size=config.model.kernel_size,
            n_channels=config.model.n_channels,
            depth=config.model.depth,
            normalize=config.model.normalize,
            gauge_fn=gauge_fn,
            constraint_fn=constraint_fn
        )
    elif name == 'BackflowCPD':
        if not isinstance(hilbert, qk.hilbert.FermionicDiscreteHilbert):
            raise ValueError("Backflow Ansatz is only implemented for fermionic systems.")
        norb = hilbert.size
        nelec = np.sum(hilbert._n_elec)
        out_trafo, total_supp_dim = get_backflow_out_transformation(
            config.model.M,
            norb,
            nelec,
            config.model.restricted,
            config.model.fixed_magnetization
        )
        if isinstance(hamiltonian, qk.operator.hamiltonian.AbInitioHamiltonianOnTheFly):
            phi = get_hf_orbitals_from_file(
                config.system,
                hilbert._n_elec,
                workdir,
                restricted=config.model.restricted, 
                fixed_magnetization=config.model.fixed_magnetization
            )
        else:
            phi = get_hf_orbitals(
                config.system,
                hamiltonian,
                restricted=config.model.restricted,      
                fixed_magnetization=config.model.fixed_magnetization
            )
        if config.model.init_fun =='normal':
            init_fun = qk.nn.initializers.normal(config.model.sigma, dtype=dtype)
            orbitals = HashableArray(phi)
        elif config.model.init_fun == 'hf':
            def init_fun(key, shape, dtype):
                epsilon = jnp.ones(shape, dtype=dtype)
                first_supp_dim = np.prod(phi.shape)
                epsilon = epsilon.at[:, : first_supp_dim, 0].set(
                    phi.flatten()
                )
                epsilon = epsilon.at[:, first_supp_dim :, 0].set(0.0)
                epsilon += jax.nn.initializers.normal(config.model.sigma, dtype=epsilon.dtype)(
                    key, shape=epsilon.shape, dtype=dtype
                )
                return epsilon
            orbitals = None
        elif config.model.init_fun == 'hf-dist':
            def init_fun(key, shape, dtype):
                phi_init = phi.flatten()
                phi_init = phi_init / config.model.M
                phi_init = np.tile(phi_init, config.model.M)
                epsilon = jnp.ones(shape, dtype=dtype)
                epsilon = epsilon.at[:, :, 0].set(
                    phi_init
                )
                epsilon += jax.nn.initializers.normal(config.model.sigma, dtype=epsilon.dtype)(
                    key, shape=epsilon.shape, dtype=dtype
                )
                return epsilon
            orbitals = None
        correction_fn = qk.models.qGPS(
            hilbert,
            total_supp_dim,
            dtype=dtype,
            init_fun=init_fun,
            out_transformation=out_trafo,
            apply_fast_update=True
        )
        ma = ma_cls(
            hilbert,
            correction_fn,
            orbitals=orbitals,
            spin_symmetry_by_structure=config.model.restricted,
            fixed_magnetization=config.model.fixed_magnetization,
            apply_fast_update=True
        )
    elif name == 'CPDBackflow':
        if not isinstance(hilbert, qk.hilbert.FermionicDiscreteHilbert):
            raise ValueError("CPD backflow Ansatz is only implemented for fermionic systems.")
        if config.model.get("exchange_cutoff", None):
            if isinstance(hamiltonian, qk.operator.hamiltonian.AbInitioHamiltonianOnTheFly):
                environments = get_top_k_orbital_indices(
                    config.system,
                    config.model.exchange_cutoff,
                    workdir
                )
                environments = HashableArray(environments)
            else:
                raise ValueError("Range cutoff is currently only supported for molecular systems.")
        else:
            environments = None
        if config.model.init_fun =='normal':
            init_fun = qk.nn.initializers.normal(config.model.sigma, dtype=dtype)
        elif config.model.init_fun == 'hf':
            if isinstance(hamiltonian, qk.operator.hamiltonian.AbInitioHamiltonianOnTheFly):
                phi = get_hf_orbitals_from_file(
                    config.system,
                    hilbert._n_elec,
                    workdir,
                    restricted=config.model.restricted, 
                    fixed_magnetization=config.model.fixed_magnetization
                )
            else:
                phi = get_hf_orbitals(
                    config.system,
                    hamiltonian,
                    restricted=config.model.restricted,      
                    fixed_magnetization=config.model.fixed_magnetization
                )
            def init_fun(key, shape, dtype):
                epsilon = jnp.ones(shape, dtype=dtype)
                epsilon = epsilon.at[:, :, :, 0, 0].set(
                    jnp.expand_dims(phi, axis=-1)
                )
                epsilon = epsilon.at[:, :, :, 1:, 0].set(0.0)
                epsilon += jax.nn.initializers.normal(config.model.sigma, dtype=epsilon.dtype)(
                    key, shape=epsilon.shape, dtype=dtype
                )
                return epsilon
        ma = ma_cls(
            hilbert,
            config.model.M,
            environments=environments,
            dtype=dtype,
            init_fun=init_fun,
            restricted=config.model.restricted,
            fixed_magnetization=config.model.fixed_magnetization
        )
    elif name == "SlaterqGPS":
        if not isinstance(hilbert, qk.hilbert.FermionicDiscreteHilbert):
            raise ValueError("SlaterqGPS Ansatz is only implemented for fermionic systems.")
        if not hilbert._n_elec[0] == hilbert._n_elec[1]:
            raise ValueError("SlaterqGPS Ansatz currently only supports restricted spin-orbitals.")
        norb = hilbert.size
        nelec = np.sum(hilbert._n_elec)
        if isinstance(hamiltonian, qk.operator.hamiltonian.AbInitioHamiltonianOnTheFly):
            phi = get_hf_orbitals_from_file(
                config.system,
                hilbert._n_elec,
                workdir,
                restricted=True, 
                fixed_magnetization=True
            )
        else:
            phi = get_hf_orbitals(
                config.system,
                hamiltonian,
                restricted=True,      
                fixed_magnetization=True
            )
        def slater_init(key, shape, dtype):
            return jnp.array(phi).astype(dtype).reshape((1, norb, nelec // 2))
        qGPS_init_fun = qk.nn.initializers.normal(config.model.sigma, dtype=dtype)
        ma = ma_cls(
            hilbert,
            dtype=dtype,
            M=config.model.M,
            slater_init_fun=slater_init,
            qGPS_init_fun=qGPS_init_fun,
            apply_fast_update=True
        )
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
    return ma

def get_symmetry_transformation_spin(name : str, translations : bool, point_symmetries: bool, spin_flip : bool, graph : AbstractGraph) -> Union[Tuple[Callable, Callable], Callable]:
    """
    Return the appropriate spin symmetry transformations
    
    Args:
        name : name of the Ansatz
        translations : whether to include translations or not
        point_symmetries : whether to include point-group symmetries or not
        spin_flip : whether to include spin_flip symmetry or not
        graph : underlying graph of the system

    Returns:
        spin symmetry transformations. For the qGPS Ansatz also the inverse transformations are returned
    """
    automorphisms = translations or point_symmetries
    if automorphisms:
        if translations and point_symmetries:
            syms = graph.automorphisms().to_array().T
        elif translations:
            syms = graph.translation_group().to_array().T
        elif point_symmetries:
            syms = graph.point_group().to_array().T
        inv_syms = np.zeros(syms.shape, dtype=syms.dtype)
        for i in range(syms.shape[0]):
            for j in range(syms.shape[1]):
                inv_syms[syms[i,j], j] = i
        syms = jnp.array(syms)
        inv_syms = jnp.array(inv_syms)
    if 'AR' in name:
        if automorphisms and spin_flip:
            def symmetries(samples : Array) -> Array:
                out = jnp.take(samples, syms, axis=-1)
                out = jnp.concatenate((out, -out), axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.concatenate((inv_syms[indices], inv_syms[indices]), axis=-1)
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1])
                inv_sym_occs = jnp.concatenate((inv_sym_occs, -inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites
        elif automorphisms:
            def symmetries(samples : Array) -> Array:
                out = jnp.take(samples, syms, axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = inv_syms[indices]
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1])
                return inv_sym_occs, inv_sym_sites
        elif spin_flip:
            def symmetries(samples : Array) -> Array:
                out = jnp.expand_dims(samples, axis=-1)
                out = jnp.concatenate((out, -out), axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_sites = jnp.concatenate((inv_sym_sites, inv_sym_sites), axis=-1)
                inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
                inv_sym_occs = jnp.concatenate((inv_sym_occs, -inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites
        else:
            def symmetries(samples : Array) -> Array:
                out = jnp.expand_dims(samples, axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
                return inv_sym_occs, inv_sym_sites
    else:
        if automorphisms and spin_flip:
            def symmetries(samples):
                out = jnp.take(samples, syms, axis=-1)
                out = jnp.concatenate((out, 1-out), axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.concatenate((inv_syms[indices], inv_syms[indices]), axis=-1)
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1])
                inv_sym_occs = jnp.concatenate((inv_sym_occs, 1-inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites
        elif automorphisms:
            def symmetries(samples):
                out = jnp.take(samples, syms, axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = inv_syms[indices]
                inv_sym_occs = jnp.tile(jnp.expand_dims(sample_at_indices, axis=-1), syms.shape[1])
                return inv_sym_occs, inv_sym_sites
        elif spin_flip:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                out = jnp.concatenate((out, 1-out), axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_sites = jnp.concatenate((inv_sym_sites, inv_sym_sites), axis=-1)
                inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
                inv_sym_occs = jnp.concatenate((inv_sym_occs, 1-inv_sym_occs), axis=-1)
                return inv_sym_occs, inv_sym_sites
        else:
            def symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                return out
            def inv_symmetries(sample_at_indices, indices):
                inv_sym_sites = jnp.expand_dims(indices, axis=-1)
                inv_sym_occs = jnp.expand_dims(sample_at_indices, axis=-1)
                return inv_sym_occs, inv_sym_sites
    return symmetries, inv_symmetries

def count_spins(spins : Array) -> Array:
    """
    Count the number of down- and up-spins in a batch of local configurations x_i,
    where x_i can be equal to:
        - 0 if it is occupied by an down-spin
        - 1 if it is occupied by a up-spin

    Args:
        spins : array of local configurations (batch,)

    Returns:
        the number of down- and up-spins for each configuration in the batch (batch, 2)
    """
    # TODO: extend to cases beyond D=2
    return jnp.stack([(spins+1)&1, ((spins+1)&2)/2], axis=-1).astype(jnp.int32)

def count_spins_fermionic(spins : Array) -> Array:
    """
    Count the spin-up and down electrons in a batch of local occupations x_i,
    where x_i can be equal to:
        - 0 if it is unoccupied
        - 1 if it is occupied by a single spin-up electron
        - 2 if it is occupied by a single spin-down electron
        - 3 if it is doubly-occupied

    Args:
        spins : array of local configurations (batch,)

    Returns:
        the number of spin-up and down electrons for each configuration in the batch (batch, 4)
    """
    zeros = jnp.zeros(spins.shape[0])
    up_spins = spins&1
    down_spins = (spins&2)/2
    return jnp.stack([zeros, up_spins, down_spins, zeros], axis=-1).astype(jnp.int32)

def renormalize_log_psi(n_spins : Array, hilbert : HomogeneousHilbert, index : int) -> Array:
    """
    Renormalize the log-amplitude to conserve the number of up- and down-spins

    Args:
        n_spins : number of up- and down-spins up to index (batch, 2)
        hilbert : Hilbert space from which configurations are sampled
        index : site index
    
    Returns:
        renormalized log-amplitude (batch,)
    """
    # TODO: extend to cases where total_sz != 0
    # FIXME: can be rewritten as
    # jnp.where(n_spins < (hilbert.size // 2), 0., -jnp.inf)
    return jnp.log(jnp.heaviside(hilbert.size//2-n_spins, 0))

@partial(jax.vmap, in_axes=(0, None, None))
def renormalize_log_psi_fermionic(n_spins : Array, hilbert : HomogeneousHilbert, index : int) -> Array:
    """
    Renormalize the log-amplitude to conserve the number of spin-up and down electrons

    Args:
        n_spins : number of spin-up and down electrons up to index (batch, 4)
        hilbert : Hilbert space from which configurations are sampled
        index : site index
    
    Returns:
        renormalized log-amplitude (batch,)
    """
    # Compute difference between spin-up (spin-down) electrons up to index and
    # total number of spin-up (spin-down) electrons
    diff = jnp.array(hilbert._n_elec, jnp.int32)-n_spins[1:3]
    
    # 1. if the number of spin-up (spin-down) electrons until index
    #    is equal to n_elec_up (n_elec_down), then set to 0 the probability
    #    of sampling a singly occupied orbital with a spin-up (spin-down)
    #    electron, as well as the probability of sampling a doubly occupied orbital
    log_psi = jnp.zeros(hilbert.local_size)
    # TODO: benchmark implementations on GPU with jax.lax.cond and gpu_cond
    log_psi = jax.lax.cond(
        diff[0] == 0,
        lambda log_psi: log_psi.at[1].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        diff[1] == 0,
        lambda log_psi: log_psi.at[2].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        (diff == 0).any(),
        lambda log_psi: log_psi.at[3].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    
    # 2. if the number of spin-up (spin-down) electrons that still need to be
    #    distributed, is smaller or equal than the number of sites left, then set the probability
    #    of sampling an empty orbital and one with the opposite spin to 0
    log_psi = jax.lax.cond(
        (diff[0] >= (hilbert.size-index)).any(),
        lambda log_psi: log_psi.at[np.array([0,2])].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    log_psi = jax.lax.cond(
        (diff[1] >= (hilbert.size-index)).any(),
        lambda log_psi: log_psi.at[np.array([0,1])].set(-jnp.inf),
        lambda log_psi: log_psi,
        log_psi
    )
    return log_psi

def get_out_transformation(name: str, apply_exp: bool):
    """
    Return the transformation of the ouput layer

    Args:
        name : name of the Ansatz
        apply_exp : whether to apply the exponential or not

    Returns:
        a callable function that is applied in the output layer of a GPS model
    """
    if name == 'qGPS' or name == 'PlaquetteqGPS':
        axis = (-2,-1)
    elif 'AR' in name:
        axis = -1
    if apply_exp:
        out_trafo = lambda x : jnp.sum(x, axis=axis)
    else:
        out_trafo = lambda x : jnp.log(jnp.sum(x, axis=axis)+0.j)
    return out_trafo

def get_plaquettes_and_masks(hilbert : HomogeneousHilbert, graph : AbstractGraph):
    """
    Return the filter plaquettes and masks for a filter-based GPS Ansatz

    Args:
        hilbert : Hilbert space on which the model should act
        graph : graph associated with the Hilbert space

    Returns:
        a tuple containing the filter plaquettes and masks for a filter-based GPS Ansatz
    """
    L = hilbert.size
    if graph and graph.ndim == 2 and graph.pbc.all():
        translations = graph.translation_group().to_array()
        plaquettes = translations[np.argsort(translations[:,0])]
        plaquettes = HashableArray(plaquettes)
    else:
        plaquettes = HashableArray(circulant(np.arange(L)))
    masks = HashableArray(np.where(plaquettes >= np.repeat([np.arange(L)], L, axis=0).T, 0, 1))
    return (plaquettes, masks)

def get_hf_orbitals(config : ConfigDict, hamiltonian : Hamiltonian, restricted: bool=True, fixed_magnetization: bool=True):
    if mpi_rank == 0:
        # Setup molecular system
        mol, h1, h2, norb, n_elec, nelec = setup_mol(config, hamiltonian)

        # Calculate the mean-field Hartree-Fock energy and wave function
        if restricted:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = ao2mo.restore(8, h2, norb)
        _, vecs = np.linalg.eigh(h1)

        # Optimize
        if not restricted:
            # Break spin-symmetry
            dm_alpha, dm_beta = mf.get_init_guess()
            dm_beta[:2, :2] = 0
            init_dens = (dm_alpha, dm_beta)
            mf = scf.newton(mf)
            mf.kernel(dm0=init_dens)
            mo1 = mf.stability(external=True)[0]
            mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo1))
            mo1 = mf.stability(external=True)[0]
            assert (mf.converged)
        else:
            # Check that orbitals are restricted
            assert (n_elec[0] == n_elec[1])
            init_dens = np.dot(vecs[:, :n_elec[0]], vecs[:, :n_elec[0]].T)
            mf.kernel(dm0=init_dens)
            if not mf.converged:
                mf = scf.newton(mf)
                mf.kernel(mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
            assert (mf.converged)
        
        # Return orbitals
        nelec = np.sum(n_elec)
        if fixed_magnetization:
            if restricted:
                orbitals = mf.mo_coeff[:, :n_elec[0]]
            else:
                mo_coeff = np.reshape(mf.mo_coeff, (2, norb, norb))
                orbitals = np.concatenate([mo_coeff[0, :, :n_elec[0]], mo_coeff[1, :, :n_elec[1]]], axis=1)
        else:
            orbitals = np.zeros((2*norb, nelec))
            mo_coeff = np.reshape(mf.mo_coeff, (2, norb, norb))
            orbitals[:norb, :n_elec[0]] = mo_coeff[0, :, :n_elec[0]]
            orbitals[norb:, n_elec[0]:] = mo_coeff[1, :, :n_elec[1]]
    else:
        orbitals = None
    orbitals = mpi_bcast(orbitals, root=0)
    return orbitals

@dispatch
def setup_mol(config: ConfigDict, hamiltonian: AbInitioHamiltonianOnTheFly):
    norb = hamiltonian.hilbert.size
    n_elec = hamiltonian.hilbert._n_elec
    nelec = np.sum(n_elec)
    mol = gto.Mole()
    mol.build(
        atom=config.molecule,
        basis=config.basis_set,
        symmetry=config.symmetry,
        unit=config.unit
    )
    h1 = hamiltonian.h_mat
    h2 = hamiltonian.eri_mat
    return mol, h1, h2, norb, n_elec, nelec

@dispatch
def setup_mol(config : ConfigDict, hamiltonian : FermiHubbardOnTheFly):
    norb = hamiltonian.hilbert.size
    n_elec = hamiltonian.hilbert._n_elec
    nelec = np.sum(n_elec)
    mol = gto.M()
    mol.nelectron = nelec
    mol.spin = n_elec[0] - n_elec[1]
    h1 = np.zeros((norb, norb))
    for i, edge in enumerate(hamiltonian.edges):
        h1[edge[0], edge[1]] = -hamiltonian.t[i]
        h1[edge[1], edge[0]] = -hamiltonian.t[i]
    h2 = np.zeros((norb, norb, norb, norb))
    np.fill_diagonal(h2, hamiltonian.U)
    return mol, h1, h2, norb, n_elec, nelec

def get_hf_orbitals_from_file(config: ConfigDict, n_elec: Tuple[int, int], workdir: str=None, restricted: bool=True, fixed_magnetization: bool=True) -> Array:
    if mpi_rank == 0:
        if workdir is None:
            workdir = os.getcwd()
        hf_orbitals_path = os.path.join(workdir, 'hf_orbitals.npy')
        if os.path.exists(hf_orbitals_path):
            hf_orbitals = np.load(hf_orbitals_path) # (norb, nelec//2) or (norb, nelec)
        else:
            raise FileNotFoundError('No HF orbitals found in workdir')
        nelec = np.sum(n_elec)
        if fixed_magnetization:
            if restricted:
                assert hf_orbitals.shape[1] == nelec//2
                orbitals = hf_orbitals
            else:
                if hf_orbitals.shape[1] == nelec//2:
                    orbitals = np.concatenate((hf_orbitals, hf_orbitals), axis=1)
                else:
                    assert hf_orbitals.shape[1] == nelec
                    orbitals = hf_orbitals
        else:
            norb = hf_orbitals.shape[0]
            if config.get('n_elec', None):
                hf_orbitals_a = hf_orbitals[:, :n_elec[0]]
                hf_orbitals_b = hf_orbitals[:, :n_elec[1]]
                orbitals = np.zeros((2*norb, nelec))
                orbitals[:norb, :n_elec[0]] = hf_orbitals_a
                orbitals[norb:, n_elec[1]:] = hf_orbitals_b
            else:
                nelec = hf_orbitals.shape[1]*2
                orbitals = np.zeros((2*norb, nelec))
                orbitals[:norb, :nelec//2] = hf_orbitals
                orbitals[norb:, nelec//2:] = hf_orbitals
    else:
        orbitals = None
    orbitals = mpi_bcast(orbitals, root=0)
    return orbitals

def get_backflow_out_transformation(M: int, norb: int, nelec: int, restricted: bool=True, fixed_magnetization: bool=True):
    """
    Return the transformation of the ouput layer for a GPS model to work within a backflow Ansatz

    Args:
        M : support dimension of each GPS backflow orbital model
        norb : number of orbitals
        nelec : number of electrons
        restricted : whether the α and β orbitals are the same or not
        fixed_magnetization : whether magnetization should be conserved or not

    Returns:
        a callable function that is applied in the output layer of a GPS model and
        the total support dimension of the GPS model
    """
    if fixed_magnetization:
        if restricted:
            shape = (M, norb, nelec//2)
        else:
            shape = (M, norb, nelec)
    else:
        shape = (M, 2*norb, nelec)
    def out_trafo(x):
        batch_size = x.shape[0]
        n_syms = x.shape[-1]
        # Reshape output into (B, M, L, N, T)
        x = jnp.reshape(x, (batch_size,)+shape+(n_syms,))
        # Sum over support dim M
        out = jnp.sum(x, axis=1)
        return out
    return out_trafo, np.prod(shape)

def get_top_k_orbital_indices(config: ConfigDict, exchange_cutoff: int, workdir: str=None) -> Array:
    if mpi_rank == 0:
        # Load exchange matrix
        if workdir is None:
            workdir = os.getcwd()
        exchange_path = os.path.join(workdir, 'exchange.npy')
        if os.path.exists(exchange_path):
            em = np.load(exchange_path) # (norb, norb)
        else:
            raise FileNotFoundError('No exchange matrix found in workdir')
        
        # Transform to a local orbital basis, if necessary
        if 'local' in config.basis:
            basis_path = os.path.join(workdir, 'basis.npy')
            if os.path.exists(basis_path):
                basis = np.load(basis_path) # (norb, norb)
            else:
                raise FileNotFoundError('No basis file found in workdir')
            em = np.linalg.multi_dot((basis.T, em, basis))

        # Generate environment matrix of top-K closest coupled orbitals for each orbital
        top_k_orbital_indices = np.flip(np.argsort(np.abs(em), axis=1)[:, -exchange_cutoff:], axis=1)
    else:
        top_k_orbital_indices = None
    top_k_orbital_indices = mpi_bcast(top_k_orbital_indices, root=0)
    return top_k_orbital_indices