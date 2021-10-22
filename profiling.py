import json
import os
import configargparse
import jax
import netket as nk
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from functools import partial
from qgps import QGPS
from arqgps import ARQGPS, FastARQGPS, FastARQGPSSymm
from autoreg import ARDirectSampler
from initializers import gaussian
from utils import create_result, dir_path, save_config, time_fn
from pprint import pprint


# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Parse arguments
parser = configargparse.ArgumentParser(
    default_config_files=['./heisenberg1d_config.yaml'],
    description='Profile different aspects of a model')
parser.add_argument('-c', '--config', is_config_file=True,
    help='Path to configuration file')
parser.add_argument('--L', type=int, default=4,
    help='Number of sites in the system (default: 4)')
parser.add_argument('--N', type=int, default=1,
    help='Bond dimension of the QGPS Ansatz (default: 1)')
parser.add_argument('--alpha', type=int, default=1,
    help='Feature density of the RBM Ansatz (default: 1)')
parser.add_argument('--scale', type=float, default=0.01,
    help='Scale of the initialized QGPS Ansatz parameters (default: 0.01)')
parser.add_argument('--maxval', type=float, default=0.1,
    help='Max value of the phase of the initialized complex QGPS Ansatz parameters (default: 0.1)')
parser.add_argument('--ansatz', default='qgps', choices=['qgps', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm', 'rbm', 'rbm-symm'],
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
parser.add_argument('--optimizer', default='sgd-sr', choices=['sgd', 'sgd-sr', 'sgd-sr-dense', 'adam'],
    help='Optimizer used for learning (default: SGD)')
parser.add_argument('--learning-rate', type=float, default=0.01,
    help='Learning rate of SGD or Adam (default: 0.01)')
parser.add_argument('--b1', type=float, default=0.9,
    help='Weight decay of 1st moment of Adam (default: 0.9)')
parser.add_argument('--b2', type=float, default=0.999,
    help='Weight decay of 2nd moment of Adam (default: 0.999)')
parser.add_argument('--diagonal-shift', type=float, default=0.1,
    help='Diagonal shift of SR (default: 0.1)')
parser.add_argument('--sr-iterative', type=bool, default=True,
    help='Whether to use an iterative method or not for SR (default: True)')
parser.add_argument('--compare-to-ed', default=False, action='store_true',
    help='Compare energy estimate to exact diagonalisation result')
msr_parser = parser.add_mutually_exclusive_group(required=False)
msr_parser.add_argument('--msr', dest='msr', action='store_true',
    help='Turns on the Marshal Sign rule')
msr_parser.add_argument('--no-msr', dest='msr', action='store_false',
    help='Turns off the Marshal Sign rule')
parser.set_defaults(msr=True)
progress_parser = parser.add_mutually_exclusive_group(required=False)
progress_parser.add_argument('--show-progress', dest='progress', action='store_true',
    help='Shows the progress bar')
progress_parser.add_argument('--hide-progress', dest='progress', action='store_false',
    help='Hides the progress bar')
parser.set_defaults(progress=True)
parser.add_argument('--save', type=dir_path,
    help='Save result to path')

parser.add_argument('--repetitions', type=int, default=1,
    help='Number of times a function is executed to profile it (default: 1)')

# Update config
config = parser.parse_args()

# Compute samples per rank
if config.samples % n_nodes != 0:
    raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
samples_per_rank = config.samples // n_nodes

# 1D Lattice
g = nk.graph.Chain(length=config.L, pbc=True)

# Hilbert space of spins on the graph
if config.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
else:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g, sign_rule=config.msr)

# Ansatz machine
if config.dtype == 'real':
    dtype = jnp.float64
elif config.dtype == 'complex':
    dtype = jnp.complex128
if config.ansatz in ['qgps', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    eps_init = gaussian(scale=config.scale, maxval=config.maxval, dtype=dtype)
if config.ansatz == 'qgps':
    ma = QGPS(N=config.N, eps_init=eps_init, dtype=dtype)
elif config.ansatz == 'arqgps':
    ma = ARQGPS(hilbert=hi, N=config.N, L=config.L,  eps_init=eps_init, dtype=dtype)
elif config.ansatz == 'arqgps-fast':
    ma = FastARQGPS(hilbert=hi, N=config.N, L=config.L, B=samples_per_rank,  eps_init=eps_init, dtype=dtype)
elif config.ansatz == 'arqgps-fast-symm':
    ma = FastARQGPSSymm(hilbert=hi, symmetries=g.automorphisms(), N=config.N, L=config.L, B=samples_per_rank,  eps_init=eps_init, dtype=dtype)
elif config.ansatz == 'rbm':
    ma = nk.models.RBM(
        alpha=config.alpha,
        use_visible_bias=False,
        use_hidden_bias=True,
        dtype=dtype,
    )
elif config.ansatz == 'rbm-symm':
    ma = nk.models.RBMSymm(
        symmetries=g.translation_group(),
        alpha=config.alpha,
        use_visible_bias=False,
        use_hidden_bias=True,
        dtype=dtype,
    )

# Sampler
if config.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
else:
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=config.chains)

# Variational state
if config.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
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

# Variational Monte Carlo driver
vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Create profile
if config.save:
    path = create_result(config.save)
    save_config(config, path)
    profile = {
        'inference': {
            'runtime': None,
        },
        'sampling': {
            'runtime': None,
        },
        'optimization': {
            'runtime': None
        }
    }

# Setup model
key = jax.random.PRNGKey(np.random.randint(0, 100))
inputs = hi.random_state(key)
params = ma.init(key, inputs)
if config.ansatz in ['rbm', 'rbm-symm']:
    print(f"Profiling {config.ansatz} with alpha={config.alpha}")
else:
    print(f"Profiling {config.ansatz} with N={config.N}")
print(f"- # {config.dtype} variational parameters: {vs.n_parameters}")
print(f"- # sampling configurations: {vs.n_samples}")

# Time inference
print(f"Inference:")
output, runtime = time_fn(ma.apply, params, inputs, repetitions=config.repetitions)
print(f"- evaluation: {runtime} seconds")
if config.save:
    profile['inference']['runtime'] = str(runtime)

# Time sampling
print(f"Sampling:")
output, runtime = time_fn(vs.sample)
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vs.sample, repetitions=config.repetitions)
print(f"- evaluation: {runtime} seconds")
if config.save:
    profile['sampling']['runtime'] = str(runtime)

# Time optimization
print("Optimization:")
output, runtime = time_fn(vmc.advance)
print(f"- compilation+evaluation: {runtime} seconds")
output, runtime = time_fn(vmc.advance, repetitions=config.repetitions)
print(f"- evaluation: {runtime} seconds")
if config.save:
    profile['optimization']['runtime'] = str(runtime)
output_fb, runtime = time_fn(vmc._forward_and_backward, repetitions=config.repetitions)
print(f"\t|--> forward_and_backward: {runtime} seconds")
if config.save:
    profile['optimization']['forward_and_backward'] = {
    'runtime': str(runtime)
    }
output_eg, runtime = time_fn(vmc.state.expect_and_grad, vmc._ham, repetitions=config.repetitions)
print(f"\t\t|--> expect_and_grad: {runtime} seconds")
if config.save:
    profile['optimization']['forward_and_backward']['expect_and_grad'] = {
        'runtime': str(runtime)
    }
output, runtime = time_fn(vmc.preconditioner, vmc.state, output_eg[1], repetitions=config.repetitions)
print(f"\t\t|--> preconditioner: {runtime} seconds")
if config.save:
    profile['optimization']['forward_and_backward']['preconditioner'] = {
        'runtime': str(runtime)
    }
output, runtime = time_fn(vmc.update_parameters, output_fb, repetitions=config.repetitions)
print(f"\t|--> update_parameters: {runtime} seconds")
if config.save:
    profile['optimization']['update_parameters'] = {
        'runtime': str(runtime)
    }

# Save profile
if config.save:
    with open(os.path.join(path, "profile.json"), "w") as f:
        json.dump(profile, f)
    print(f"\nProfile saved at {path}")
