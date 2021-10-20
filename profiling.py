import argparse
import jax
import netket as nk
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from qgps import FastQGPS, QGPS
from arqgps import ARQGPS, FastARQGPS, FastARQGPSSymm
from autoreg import ARDirectSampler
from initializers import gaussian
from utils import time_fn


# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Parse arguments
parser = argparse.ArgumentParser(
    description='Profile Ansatz on a Heisenberg 1D model')
parser.add_argument('-L', type=int, default=4,
    help='Number of sites in the system (default: 4)')
parser.add_argument('-N', type=int, default=1,
    help='Bond dimension of the QGPS Ansatz (default: 1)')
parser.add_argument('--alpha', type=int, default=4,
    help='Alpha parameters of RBM Ansatz (default: 4)')
parser.add_argument('--ansatz', default='qgps', choices=['qgps', 'qgps-fast', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm', 'rbm', 'rbm-symm', 'mps'],
    help='Ansatz for the wavefunction (default: qgps)')
parser.add_argument('--dtype', default='real', choices=['real', 'complex'],
    help='Type of the Ansatz parameters (default: real')
parser.add_argument('--samples', type=int, default=1000,
    help='Number of samples used in VMC (default: 1000)')
parser.add_argument('--repetitions', type=int, default=1,
    help='Number of times the function is executed to compute a runtime (default: 1)')
args = parser.parse_args()

# Compute samples per rank
if args.samples % n_nodes != 0:
    raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
samples_per_rank = args.samples // n_nodes

# 1D Lattice
g = nk.graph.Chain(length=args.L, pbc=True)

# Hilbert space of spins on the graph
if args.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
else:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g, sign_rule=False)

# Ansatz machine
if args.dtype == 'real':
    dtype = jnp.float64
elif args.dtype == 'complex':
    dtype = jnp.complex64
if args.ansatz in ['qgps', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    eps_init = gaussian(scale=0.001, maxval=0.1, dtype=dtype)
if args.ansatz == 'qgps':
    ma = QGPS(N=args.N, dtype=dtype)
elif args.ansatz == 'qgps-fast':
    ma = FastQGPS(N=args.N, dtype=dtype)
elif args.ansatz == 'arqgps':
    ma = ARQGPS(hilbert=hi, N=args.N, L=args.L, dtype=dtype)
elif args.ansatz == 'arqgps-fast':
    ma = FastARQGPS(hilbert=hi, N=args.N, L=args.L, B=samples_per_rank, dtype=dtype)
elif args.ansatz == 'arqgps-fast-symm':
    ma = FastARQGPSSymm(hilbert=hi, symmetries=g.automorphisms(), N=args.N, L=args.L, B=samples_per_rank,  eps_init=eps_init, dtype=dtype)
elif args.ansatz == 'rbm':
    ma = nk.models.RBM(
        alpha=args.alpha,
        use_visible_bias=False,
        use_hidden_bias=True,
        dtype=dtype,
    )
elif args.ansatz == 'rbm-symm':
    ma = nk.models.RBMSymm(
        symmetries=g.translation_group(),
        alpha=args.alpha,
        use_visible_bias=False,
        use_hidden_bias=True,
        dtype=dtype,
    )
elif args.ansatz == 'mps':
    ma = nk.models.MPSPeriodic(
        hilbert=hi,
        graph=g,
        bond_dim=args.N,
        dtype=dtype
    )

# Sampler
if args.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
else:
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=1)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

# VMC state
if args.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    vs = nk.vqs.MCState(sa, ma, n_samples=args.samples)
else:
    vs = nk.vqs.MCState(sa, ma, n_samples=args.samples, n_discard_per_chain=int(args.samples/10))
vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Setup model
key = jax.random.PRNGKey(np.random.randint(0, 100))
inputs = hi.random_state(key)
params = ma.init(key, inputs)

# Time model call
print(f"Model call ({vs.n_parameters} {args.dtype} parameters):")
output, runtime = time_fn(ma.apply, params, inputs, repetitions=args.repetitions)
print(f"- evaluation: {runtime} seconds")

# Time sampling
print(f"Sampling {vs.n_samples} configurations:")
output, runtime = time_fn(vs.sample)
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vs.sample, repetitions=args.repetitions)
print(f"- evaluation: {runtime} seconds")

# Time energy and grad
print("Energy and grad:")
output, runtime = time_fn(vs.expect_and_grad, ha.collect())
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vs.expect_and_grad, ha.collect(), repetitions=args.repetitions)
print(f"- evaluation: {runtime} seconds")

# Time optimization
print("Optimization:")
output, runtime = time_fn(vmc.advance)
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vmc.advance, repetitions=args.repetitions)
print(f"- evaluation: {runtime} seconds")
output, runtime = time_fn(vmc._forward_and_backward, repetitions=args.repetitions)
print(f"\t|--> forward_and_backward: {runtime} seconds")
output, runtime = time_fn(vmc.update_parameters, output, repetitions=args.repetitions)
print(f"\t|--> update_parameters: {runtime} seconds")
