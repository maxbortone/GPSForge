import numpy as np
import jax.numpy as jnp
import netket as nk
import qGPSKet as qk
from mpi4py import MPI
from functools import partial
from dataclasses import dataclass


# MPI variables
@dataclass
class MPIVariables:
    comm : MPI.Comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
    rank : int = comm.Get_rank()
    n_nodes : int = comm.Get_size()

MPIVars = MPIVariables()

def compute_chunk_size(multiplier, n_samples, size):
    if multiplier > 0:
        chunk_size = int(2**(np.ceil(np.log2(n_samples*size*multiplier))))
    else:
        raise ValueError("Chunk size multiplier needs to be > 0.0")
    return chunk_size

def setup_vmc(config):
    # System
    if config.constrained:
        total_sz = 0.
    else:
        total_sz = None
    ha = qk.operator.hamiltonian.get_J1_J2_Hamiltonian(config.Lx, Ly=config.Ly, J1=config.J1, J2=config.J2, sign_rule=config.sign_rule, total_sz=total_sz)
    hi = ha.hilbert
    g = ha.graph

    # Ansatz model
    if config.dtype == 'real':
        dtype = jnp.float64
    elif config.dtype == 'complex':
        dtype = jnp.complex128
    init_fun = qk.nn.initializers.normal(sigma=config.sigma, dtype=dtype)
    to_indices = lambda x: jnp.asarray((x+hi.local_size-1)/hi.local_size, jnp.int8)
    if isinstance(hi, nk.hilbert.Spin):
        if config.symmetries == 'all':
            automorphisms = True
            spin_flip = True
        elif config.symmetries == 'none':
            automorphisms = False
            spin_flip = False
        elif config.symmetries == 'automorphisms':
            automorphisms = True
            spin_flip = False
        elif config.symmetries == 'spin-flip':
            automorphisms = False
            spin_flip = True
        symmetries, inv_symmetries = qk.models.get_sym_transformation_spin(g, automorphisms=automorphisms, spin_flip=spin_flip)
    if config.ansatz == 'qgps':
        ma = qk.models.qGPS(
            hi, config.M,
            dtype=dtype, init_fun=init_fun,
            to_indices=to_indices, syms=(symmetries, inv_symmetries))
    elif config.ansatz == 'arqgps':
        ma = qk.models.ARqGPS(
            hi, config.M,
            dtype=dtype, init_fun=init_fun,
            to_indices=to_indices,
            apply_symmetries=symmetries)

    # Compute samples per rank
    if config.samples % MPIVars.n_nodes != 0:
        raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
    samples_per_rank = config.samples // MPIVars.n_nodes

    # Sampler
    if config.sampler == 'ar-direct' and config.ansatz == 'arqgps':
        sa = qk.sampler.ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
    elif config.sampler == 'metropolis-exchange':
        sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=config.chains, n_sweeps=config.sweeps, d_max=config.Lx//2)
    elif config.sampler == 'metropolis-local':
        sa = nk.sampler.MetropolisLocal(hi, n_chains_per_rank=config.chains, n_sweeps=config.sweeps)

    # Variational state
    if sa.is_exact:
        vs = nk.vqs.MCState(sa, ma, n_samples=config.samples)
    else:
        vs = nk.vqs.MCState(sa, ma, n_samples=config.samples, n_discard_per_chain=config.discard)

    # Optimizer
    if config.optimizer == 'sgd':
        op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
        sr = None
    elif config.optimizer == 'sgd-sr':
        op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
        sr = nk.optimizer.SR(diag_shift=config.diagonal_shift, iterative=config.sr_iterative)
    elif config.optimizer == 'sgd-sr-dense':
        op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
        sr = nk.optimizer.SR(qgt=partial(nk.optimizer.qgt.QGTJacobianDense, mode=config.dtype, diag_shift=config.diagonal_shift), iterative=config.sr_iterative)
    elif config.optimizer == 'adam':
        op = nk.optimizer.Adam(learning_rate=config.learning_rate, b1=config.b1, b2=config.b2)
        sr = None

    return ha, (op, sr), vs