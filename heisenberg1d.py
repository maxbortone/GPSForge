import os
import argparse
import netket as nk
import scipy
import jax.numpy as jnp
from qgps import QGPS
from arqgps import ARQGPS
from utils import dir_path


# Parse arguments
parser = argparse.ArgumentParser(
    description='Estimate energy of Heisenberg 1D model')
parser.add_argument('-L', type=int, default=4,
    help='Number of sites in the system (default: 4)')
parser.add_argument('-N', type=int, default=1,
    help='Bond dimension of the QGPS Ansatz (default: 1)')
parser.add_argument('--ansatz', default='qgps', choices=['qgps', 'ar-qgps', 'rbm', 'rbm-symm'],
    help='Ansatz for the wavefunction (default: qgps)')
parser.add_argument('--dtype', default='real', choices=['real', 'complex'],
    help='Type of the Ansatz parameters (default: real')
parser.add_argument('--samples', type=int, default=1000,
    help='Number of samples used in VMC (default: 1000)')
parser.add_argument('--discard', type=int, default=100,
    help='Number of samples discarded during VMC (default: 100)')
parser.add_argument('--chains', type=int, default=1,
    help='Number of chains used in VMC (default: 1)')
parser.add_argument('--iterations', type=int, default=100,
    help='Number of VMC iterations (default: 100)')
parser.add_argument('--lr', type=float, default=0.01,
    help='Learning rate of SGD (default: 0.01)')
parser.add_argument('--ds', type=float, default=0.1,
    help='Diagonal shift of SR (default: 0.1)')
parser.add_argument('--msr', default=True, action=argparse.BooleanOptionalAction,
    help='Turn on the Marshal Sign rule (default: True')
parser.add_argument('--save', type=dir_path,
    help='Save result to path')
args = parser.parse_args()

# 1D Lattice
g = nk.graph.Chain(length=args.L, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g, sign_rule=args.msr)

# Ansatz machine
if args.dtype == 'real':
    dtype = jnp.float64
elif args.dtype == 'complex':
    dtype = jnp.complex64
if args.ansatz == 'qgps':
    ma = QGPS(N=args.N, dtype=dtype)
elif args.ansatz == 'ar-qgps':
    ma = ARQGPS(N=args.N, L=args.L, dtype=dtype)
elif args.ansatz == 'rbm':
    ma = nk.models.RBM(
        alpha=4,
        use_visible_bias=False,
        use_hidden_bias=True,
        dtype=dtype,
    )
elif args.ansatz == 'rbm-symm':
    ma = nk.models.RBMSymm(
        symmetries=g.translation_group(),
        alpha=4,
        use_visible_bias=False,
        use_hidden_bias=True,
        dtype=dtype,
    )

# Metropolis Local Sampling
if args.ansatz == 'ar-qgps':
    sa = nk.sampler.ARDirectSampler(hi)
else:
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=args.chains)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=args.lr)
sr = nk.optimizer.SR(diag_shift=args.ds)

# Variational Monte Carlo driver
if args.ansatz == 'ar-qgps':
    vs = nk.vqs.MCState(sa, ma, n_samples=args.samples)
else:
    vs = nk.vqs.MCState(sa, ma, n_samples=args.samples, n_discard_per_chain=args.discard)
vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Print parameter structure
msr_status = 'on' if args.msr else 'off'
print(f"Running optimisation of {args.ansatz} with N={args.N}")
print(f"- # {args.dtype} variational parameters: {vs.n_parameters}")
print(f"- MSR: {msr_status}")

# Run the optimization for 300 iterations
if args.save:
    path = os.path.join(args.save, f"heisenberg1d_L{args.L}_N{args.N}_{args.ansatz}_{args.dtype}")
    vmc.run(n_iter=args.iterations, out=path)
else:
    for it in vmc.iter(args.iterations, 10):
        print(it, vmc.energy)

exact_ens = scipy.sparse.linalg.eigsh(ha.to_sparse(),k=1,which='SA',return_eigenvectors=False)
print(f"Estimated energy is: {vmc.energy.mean}")
print(f"Exact energy is: {exact_ens[0]}")
print(f"Relative error is: {abs((vmc.energy.mean-exact_ens[0])/exact_ens[0])}")