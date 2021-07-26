import argparse
from autoreg import ARDirectSampler
import jax
import netket as nk
import jax.numpy as jnp
from qgps import FastQGPS, QGPS
from arqgps import ARQGPS, FastARQGPS
from timeit import default_timer as timer
from datetime import timedelta


def time_fn(fn, *args):
    start = timer()
    output = fn(*args)
    jax.tree_map(lambda x: x.block_until_ready(), output)
    end = timer()
    runtime = timedelta(seconds=end-start)
    return output, runtime

# Parse arguments
parser = argparse.ArgumentParser(
    description='Profile Ansatz on a Heisenberg 1D model')
parser.add_argument('-L', type=int, default=4,
    help='Number of sites in the system (default: 4)')
parser.add_argument('-N', type=int, default=1,
    help='Bond dimension of the QGPS Ansatz (default: 1)')
parser.add_argument('--alpha', type=int, default=4,
    help='Alpha parameters of RBM Ansatz (default: 4)')
parser.add_argument('--ansatz', default='qgps', choices=['qgps', 'qgps-fast', 'arqgps', 'arqgps-fast', 'rbm', 'rbm-symm', 'mps'],
    help='Ansatz for the wavefunction (default: qgps)')
parser.add_argument('--dtype', default='real', choices=['real', 'complex'],
    help='Type of the Ansatz parameters (default: real')
parser.add_argument('--samples', type=int, default=1000,
    help='Number of samples used in VMC (default: 1000)')
args = parser.parse_args()

# 1D Lattice
g = nk.graph.Chain(length=args.L, pbc=True)

# Hilbert space of spins on the graph
if args.ansatz in ['arqgps', 'arqgps-fast']:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
else:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g, sign_rule=False)

# Ansatz machine
if args.dtype == 'real':
    dtype = jnp.float32
elif args.dtype == 'complex':
    dtype = jnp.complex64
if args.ansatz == 'qgps':
    ma = QGPS(N=args.N, dtype=dtype)
elif args.ansatz == 'qgps-fast':
    ma = FastQGPS(N=args.N, dtype=dtype)
elif args.ansatz == 'arqgps':
    ma = ARQGPS(hilbert=hi, N=args.N, L=args.L, dtype=dtype)
elif args.ansatz == 'arqgps-fast':
    ma = FastARQGPS(hilbert=hi, N=args.N, L=args.L, B=args.samples, dtype=dtype)
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
if args.ansatz in ['arqgps', 'arqgps-fast']:
    sa = ARDirectSampler(hi, n_chains_per_rank=args.samples)
else:
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=1)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

# VMC state
if args.ansatz in ['arqgps', 'arqgps-fast']:
    vs = nk.vqs.MCState(sa, ma, n_samples=args.samples)
else:
    vs = nk.vqs.MCState(sa, ma, n_samples=args.samples, n_discard_per_chain=int(args.samples/10))
vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)


# Setup model
key = jax.random.PRNGKey(42)
x = jax.random.choice(key, jnp.array([-1, 1]), (1, args.L,))
params = ma.init(key, x)
n_params = nk.jax.tree_size(params['params'])
print(f"Model has {n_params} {args.dtype} parameters")

# Time model call
# print("Model call:")
# output, runtime = time_fn(ma.apply, params, x)
# print(f"- evaluation: {runtime} seconds")

# Time sampling
print("Sampling:")
output, runtime = time_fn(vs.sample)
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vs.sample)
print(f"- evaluation: {runtime} seconds")

# Time energy and grad
print("Energy and grad:")
output, runtime = time_fn(vs.expect_and_grad, ha.collect())
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vs.expect_and_grad, ha.collect())
print(f"- evaluation: {runtime} seconds")

# Time optimization
print("Optimization:")
output, runtime = time_fn(vmc.advance)
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vmc.advance)
print(f"- evaluation: {runtime} seconds")
