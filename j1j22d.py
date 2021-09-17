import os
import configargparse
import netket as nk
import scipy
import numpy as np
import jax.numpy as jnp
import pandas as pd
from mpi4py import MPI
from initializers import gaussian
from qgps import QGPS
from arqgps import ARQGPS, FastARQGPS, FastARQGPSSymm
from autoreg import ARDirectSampler
from utils import create_result, dir_path, save_config


# MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parse arguments
parser = configargparse.ArgumentParser(
    default_config_files=['./j1j22d_config.yaml'],
    description='Estimate energy of J1-J2 2D model')
parser.add_argument('-c', '--config', is_config_file=True,
    help='Path to configuration file')
parser.add_argument('--L', type=int, default=6,
    help='Number of sites in one dimension of the system (default: 6)')
parser.add_argument('--J1', type=float, default=1.0,
    help='Nearest neighbor coupling (default: 1.0)')
parser.add_argument('--J2', type=float, default=0.2,
    help='Next-nearest neighbor coupling (default: 0.2)')
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
parser.add_argument('--learning-rate', type=float, default=0.01,
    help='Learning rate of SGD (default: 0.01)')
parser.add_argument('--diagonal-shift', type=float, default=0.1,
    help='Diagonal shift of SR (default: 0.1)')
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

# 2D Lattice
L = config.L
edge_colors = []
for i in range(L):
    for j in range(L):
        edge_colors.append([i*L+j, i*L+(j+1)%L, 1]) # right nearest neighbor
        edge_colors.append([i*L+j, ((i+1)%L)*L+j, 1]) # bottom nearest neighbor
        edge_colors.append([i*L+j, ((i+1)%L)*L+(j+1)%L, 2]) # bottom-right next-nearest neighbor
        edge_colors.append([i*L+j, ((i+1)%L)*L+(j-1)%L, 2]) # bottom-left next-nearest neighbor
g = nk.graph.Graph(edges=edge_colors)

# Hilbert space of spins on the graph
if config.ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
else:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# J1-J2 spin hamiltonian
J1 = config.J1
J2 = config.J2
msr_rot = -1.0 if config.msr else 1.0
sigmaz = [[1, 0], [0, -1]]
mszsz = (np.kron(sigmaz, sigmaz))
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
bond_operator = [
    (msr_rot*J1*mszsz).tolist(),
    (J2*mszsz).tolist(),
    (msr_rot*J1*exchange).tolist(),
    (J2*exchange).tolist(),

]
bond_color = [1, 2, 1, 2]
ha = nk.operator.GraphOperator(hilbert=hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)

# Ansatz machine
if config.dtype == 'real':
    dtype = jnp.float64
elif config.dtype == 'complex':
    dtype = jnp.complex64
if config.ansatz in ['qgps', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
    eps_init = gaussian(scale=config.scale, maxval=config.maxval, dtype=dtype)
if config.ansatz == 'qgps':
    ma = QGPS(N=config.N, eps_init=eps_init, dtype=dtype)
elif config.ansatz == 'arqgps':
    ma = ARQGPS(hilbert=hi, N=config.N, L=config.L**2,  eps_init=eps_init, dtype=dtype)
elif config.ansatz == 'arqgps-fast':
    ma = FastARQGPS(hilbert=hi, N=config.N, L=config.L**2, B=config.samples,  eps_init=eps_init, dtype=dtype)
elif config.ansatz == 'arqgps-fast-symm':
    ma = FastARQGPSSymm(hilbert=hi, symmetries=g.automorphisms(), N=config.N, L=config.L**2, B=config.samples,  eps_init=eps_init, dtype=dtype)
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
    sa = ARDirectSampler(hi, n_chains_per_rank=config.samples)
else:
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=config.chains, d_max=2)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
sr = nk.optimizer.SR(diag_shift=config.diagonal_shift)

# Variational Monte Carlo driver
if config.ansatz in ['arqgps', 'arqgps-fast']:
    vs = nk.vqs.MCState(sa, ma, n_samples=config.samples)
else:
    vs = nk.vqs.MCState(sa, ma, n_samples=config.samples, n_discard_per_chain=config.discard)
vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Print parameter structure
msr_status = 'on' if config.msr else 'off'
if rank == 0:
    if config.ansatz in ['rbm', 'rbm-symm']:
        print(f"Running optimisation of {config.ansatz} with alpha={config.alpha}")
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
        print("Iteration\t Energy statistics\t Gradient norm")
    for it in vmc.iter(config.iterations, 10):
        if rank == 0:
            if config.ansatz in ['rbm', 'rbm-symm']:
                print(f"[{it+10}/{config.iterations}] E: {vmc.energy}, ||∇E||: {np.linalg.norm(vmc._loss_grad['Dense']['kernel'])}")
            else:
                print(f"[{it+10}/{config.iterations}] E: {vmc.energy}, ||∇E||: {np.linalg.norm(vmc._loss_grad['epsilon'])}")

if config.compare_to_ed:
    # Get converged energy estimate
    # TODO: devise a better way of obtaining this,
    # as now it considers the last energy as the converged one
    estimated_energy = vs.expect(ha)

    # Get exact energy
    # base_path = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(base_path, 'result_DMRG_Heisenberg_1D.csv')
    # df = pd.read_csv(path, dtype={'L': np.int64, 'E': np.float32})
    # if (df['L']==config.L).any():
    #     exact_energy = 4*df.loc[df['L']==config.L]['E'].values[0]
    # else:
    #     exact_energy = scipy.sparse.linalg.eigsh(ha.to_sparse(),k=1,which='SA',return_eigenvectors=False)[0]
    exact_energy = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
    if rank == 0:
        print(f"Estimated energy is: {estimated_energy}")
        print(f"Exact energy is: {exact_energy}")
        print(f"Relative error is: {abs((estimated_energy.mean-exact_energy)/exact_energy)}")