import os
import configargparse
import netket as nk
import scipy
import numpy as np
import jax.numpy as jnp
from mpi4py import MPI
from functools import partial
from qgps import QGPS
from arqgps import ARQGPS, FastARQGPS, FastARQGPSSymm
from autoreg import ARDirectSampler
from initializers import gaussian
from utils import create_result, dir_path, get_exact_energy, parse_int_or_iterable, save_config


# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Parse arguments
parser = configargparse.ArgumentParser(
    default_config_files=['./heisenberg1d_config.yaml'],
    description='Estimate energy of Heisenberg 1D model')
parser.add_argument('-c', '--config', is_config_file=True,
    help='Path to configuration file')
parser.add_argument('--L', type=int, default=4,
    help='Number of sites in the system (default: 4)')
parser.add_argument('--N', type=int, default=1,
    help='Bond dimension of the QGPS Ansatz (default: 1)')
parser.add_argument('--alpha', type=int, default=1,
    help='Feature density of the RBM Ansatz (default: 1)')
parser.add_argument('--layers', type=int, default=1,
    help='Number of layers in the ARNN Ansatz (default: 1)')
parser.add_argument('--features', type=parse_int_or_iterable, default=4,
    help='Number of features per layer in the ARNN Ansatz (default: 4)')
parser.add_argument('--kernel-size', type=int, default=1,
    help='Kernel size of the ARNNConv Ansatz (default: 1)')
parser.add_argument('--kernel-dilation', type=int, default=1,
    help='Kernel dilation of the ARNNConv Ansatz (default: 1)')
parser.add_argument('--scale', type=float, default=0.01,
    help='Scale of the initialized QGPS Ansatz parameters (default: 0.01)')
parser.add_argument('--maxval', type=float, default=0.1,
    help='Max value of the phase of the initialized complex QGPS Ansatz parameters (default: 0.1)')
parser.add_argument('--ansatz', default='qgps',
    choices=['qgps', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm', 'rbm', 'rbm-symm', 'arnn-dense', 'arnn-conv1d', 'arnn-conv2d'],
    help='Ansatz for the wavefunction (default: qgps)')
parser.add_argument('--dtype', default='real', choices=['real', 'complex'],
    help='Type of the Ansatz parameters (default: real')
parser.add_argument('--constrained', type=bool, default=False,
    help='Whether to constrain the Hilbert space to the zero magnetization sector (default: False)')
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

# Update config
config = parser.parse_args()

# Compute samples per rank
if config.samples % n_nodes != 0:
    raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
samples_per_rank = config.samples // n_nodes

# 1D Lattice
g = nk.graph.Chain(length=config.L, pbc=True)

# Hilbert space of spins on the graph
if config.constrained:
    if config.ansatz in ['arnn-dense', 'arnn-conv1d', 'arnn-conv2d']:
        raise ValueError("ARNN Ansatze do not support constrained Hilbert spaces")
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)
else:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

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
elif config.ansatz == 'arnn-dense':
    ma = nk.models.ARNNDense(
        hilbert=hi,
        layers=config.layers,
        features=config.features,
        dtype=dtype
    )
elif config.ansatz == 'arnn-conv1d':
    ma = nk.models.FastARNNConv1D(
        hilbert=hi,
        layers=config.layers,
        features=config.features,
        kernel_size=config.kernel_size,
        kernel_dilation=config.kernel_dilation,
        dtype=dtype
    )

# Sampler
if config.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
elif config.ansatz in ['arnn-dense', 'arnn-conv1d', 'arnn-conv2d']:
    sa = nk.sampler.ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
else:
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=config.chains)

# Variational state
if config.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm', 'arnn-dense', 'arnn-conv1d', 'arnn-conv2d']:
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

# Print parameter structure
msr_status = 'on' if config.msr else 'off'
if rank == 0:
    if config.ansatz in ['rbm', 'rbm-symm']:
        print(f"Running optimisation of {config.ansatz} with alpha={config.alpha}")
    elif config.ansatz in ['arnn-dense', 'arnn-conv1d', 'arnn-conv2d']:
        print(f"Running optimisation of {config.ansatz} with layers={config.layers}")
    else:
        print(f"Running optimisation of {config.ansatz} with N={config.N}")
    print(f"- # {config.dtype} variational parameters: {vs.n_parameters}")
    print(f"- MSR: {msr_status}")

# Run the optimization
if config.save:
    if rank == 0:
        path = create_result(config.save)
        save_config(config, path)
    else:
        path = None
    path = comm.bcast(path, root=0)
    logger = nk.logging.JsonLog(os.path.join(path, "output"))
    vmc.run(n_iter=config.iterations, out=logger, show_progress=config.progress)
else:
    if rank == 0:
        print("Iteration\t Energy statistics\t\t Gradient norm")
    for it in vmc.iter(config.iterations, 10):
        if rank == 0:
            grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
            grad_norm = np.linalg.norm(grad)
            print(f"[{it+10}/{config.iterations}] E: {vmc.energy}, ||∇E||: {grad_norm}")

if config.compare_to_ed:
    # Get converged energy estimate
    # TODO: devise a better way of obtaining this,
    # as now it considers the last energy as the converged one
    estimated_energy = vs.expect(ha)

    # Get exact energy
    exact_energy = get_exact_energy("heisenberg1d", config)
    if exact_energy is None:
        exact_energy = scipy.sparse.linalg.eigsh(ha.to_sparse(),k=1,which='SA',return_eigenvectors=False)[0]
    if rank == 0:
        print(f"\nEstimated energy:\t{estimated_energy}")
        print(f"Exact energy:\t\t{exact_energy}")
        print(f"Relative error:\t\t{abs((estimated_energy.mean-exact_energy)/exact_energy)}")
if config.save:
    if rank == 0:
        print(f"\nResult saved at {path}")
