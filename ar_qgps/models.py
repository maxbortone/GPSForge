import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import qGPSKet as qk
from scipy.linalg import circulant
from functools import partial
from flax import linen as nn
from netket.hilbert import HomogeneousHilbert
from netket.graph import AbstractGraph
from netket.utils import HashableArray
from netket.utils.types import Array
from ml_collections import ConfigDict
from typing import Union, Tuple, Callable, Optional


def get_model(name : str, config : ConfigDict, hilbert : HomogeneousHilbert, graph : Optional[AbstractGraph]=None) -> nn.Module:
    """
    Return the model for a wavefunction Ansatz

    Args:
        name : wavefunction Ansatz name
        config : model configuration dictionary
        hilbert : Hilbert space on which the model should act
        graph : graph associated with the Hilbert space (optional)

    Returns:
        the model for the wavefunction Ansatz
    """
    if config.dtype == 'real':
        dtype = jnp.float64
    elif config.dtype == 'complex':
        dtype = jnp.complex128
    init_fn = qk.nn.initializers.normal(sigma=config.sigma, dtype=dtype)
    if isinstance(hilbert, nk.hilbert.Spin):
        # Maps [-1, 1] to [0, 1]
        to_indices_fn = lambda x: jnp.asarray((x+hilbert.local_size-1)/2, jnp.int8)
    else:
        to_indices_fn = lambda x: x.astype(jnp.uint8)
    if graph:
        symmetries_fn, inv_symmetries_fn = get_symmetry_transformation_spin(name, config, graph)
    else:
        symmetries_fn, inv_symmetries_fn = qk.models.no_syms()
    out_trafo = get_out_transformation(name, config.apply_exp)
    if name == 'qGPS':
        ma = qk.models.qGPS(
            hilbert, config.M,
            dtype=dtype,
            init_fun=init_fn,
            to_indices=to_indices_fn,
            syms=(symmetries_fn, inv_symmetries_fn),
            out_transformation=out_trafo)
    elif 'AR' in name:
        if isinstance(hilbert, nk.hilbert.Spin):
            count_spins_fn = count_spins
            renormalize_log_psi_fn = renormalize_log_psi
        elif isinstance(hilbert, qk.hilbert.FermionicDiscreteHilbert):
            count_spins_fn = count_spins_fermionic
            renormalize_log_psi_fn = renormalize_log_psi_fermionic
        ma_cls = {
            'ARqGPS': qk.models.ARqGPS,
            'ARqGPSFull': qk.models.ARqGPSFull,
            'ARPlaquetteqGPS': qk.models.ARPlaquetteqGPS
        }[name]
        args = [hilbert, config.M]
        if 'Plaquette' in name:
            args.extend(get_plaquettes_and_masks(hilbert, graph))
        ma = ma_cls(
                *args,
                dtype=dtype, init_fun=init_fn,
                to_indices=to_indices_fn,
                apply_symmetries=symmetries_fn,
                count_spins=count_spins_fn,
                renormalize_log_psi=renormalize_log_psi_fn,
                out_transformation=out_trafo)
    return ma

def get_symmetry_transformation_spin(name : str, config : ConfigDict, graph : AbstractGraph) -> Union[Tuple[Callable, Callable], Callable]:
    """
    Return the appropriate spin symmetry transformations
    
    Args:
        name : name of the model Ansatz
        config : model configuration dictionary
        graph : underlying graph of the system

    Returns:
        spin symmetry transformations. For the qGPS Ansatz also the inverse transformations are returned
    """
    automorphisms = config.symmetries in ['all', 'automorphisms']
    spin_flip = config.symmetries in ['all', 'spin-flip']
    if name == 'qGPS':
        return qk.models.get_sym_transformation_spin(graph, automorphisms, spin_flip)
    elif 'AR' in name:
        symmetries = graph.automorphisms().to_array().T
        if automorphisms and spin_flip:
            def apply_symmetries(samples : Array) -> Array:
                out = jnp.take(samples, symmetries, axis=-1)
                out = jnp.concatenate((out, -out), axis=-1)
                return out
        elif automorphisms:
            def apply_symmetries(samples : Array) -> Array:
                out = jnp.take(samples, symmetries, axis=-1)
                return out
        elif spin_flip:
            def apply_symmetries(samples : Array) -> Array:
                out = jnp.expand_dims(samples, axis=-1)
                out = jnp.concatenate((out, -out), axis=-1)
                return out
        else:
            def apply_symmetries(samples : Array) -> Array:
                out = jnp.expand_dims(samples, axis=-1)
                return out
        return apply_symmetries, None

def count_spins(spins : Array) -> Array:
    """
    Count the number of up- and down-spins in a batch of local configurations x_i,
    where x_i can be equal to:
        - 0 if it is occupied by an up-spin
        - 1 if it is occupied by a down-spin

    Args:
        spins : array of local configurations (batch,)

    Returns:
        the number of up- and down-spins for each configuration in the batch (batch, 2)
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
    if name == 'qGPS':
        axis = (-2,-1)
    elif 'AR' in name:
        axis = -1
    if apply_exp:
        out_trafo = lambda x : jnp.sum(x, axis=axis)
    else:
        out_trafo = lambda x : jnp.log(jnp.sum(x, axis=axis)+0.j)
    return out_trafo

def get_plaquettes_and_masks(hilbert : HomogeneousHilbert, graph : AbstractGraph):
    L = hilbert.size
    if graph and graph.ndim == 2:
        # TODO: maybe replace this code with a double for loop over lattice coordinates
        translations = graph.translation_group().to_array()
        plaquettes = translations[np.argsort(translations[:,0])]
        plaquettes = HashableArray(plaquettes)
    else:
        plaquettes = HashableArray(circulant(np.arange(L)))
    masks = HashableArray(np.where(plaquettes >= np.repeat([np.arange(L)], L, axis=0).T, 0, 1))
    return (plaquettes, masks)