import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from mpi4py import MPI
from arqgps import FastARQGPS, FastARQGPSSymm
from autoreg import ARDirectSampler
from initializers import gaussian


# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Model variables
key = jax.random.PRNGKey(np.random.randint(0, 100))
L = 8
N = 2
eps_init = gaussian(scale=0.01)
n_samples = 20

# Compute samples per rank
if n_samples % n_nodes != 0:
    raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
samples_per_rank = n_samples // n_nodes

# Setup
g = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
ha = nk.operator.Heisenberg(hilbert=hi, graph=g, sign_rule=False)

# Test #1
ma = FastARQGPS(hilbert=hi, N=N, L=L, B=samples_per_rank, eps_init=eps_init, dtype=jnp.complex64)
sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
samples = vs.sample()
if rank == 0:
    print("FastARQGPS:")
    print(f"- sampler.n_chains_per_rank = {sa.n_chains_per_rank}")
    print(f"- vqs.n_samples = {vs.n_samples}")
    print(f"- vqs.chain_length = {vs.chain_length}")
    print(f"- samples.shape = {samples.shape}")
np.testing.assert_equal(samples.shape, (1, samples_per_rank, L))

# Test #2
symmetries = g.automorphisms()
ma = FastARQGPSSymm(hilbert=hi, symmetries=symmetries, N=N, L=L, B=samples_per_rank, eps_init=eps_init, dtype=jnp.complex64)
sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
samples = vs.sample()
if rank == 0:
    print("FastARQGPSSymm:")
    print(f"- sampler.n_chains_per_rank = {sa.n_chains_per_rank}")
    print(f"- vqs.n_samples = {vs.n_samples}")
    print(f"- vqs.chain_length = {vs.chain_length}")
    print(f"- samples.shape = {samples.shape}")
np.testing.assert_equal(samples.shape, (1, samples_per_rank, L))

