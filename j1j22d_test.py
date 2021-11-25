import os
import json
import argparse
import jax.numpy as jnp
import netket as nk
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from arqgps import FastARQGPSSymm
from autoreg import ARDirectSampler
from initializers import gaussian
from utils import dir_path, parse_int_or_iterable, read_config, restore_model


# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Parse arguments
parser = argparse.ArgumentParser(
    description='Test the energy of a wavefunction trained on the J1-J2 2D model for different sample sizes')
parser.add_argument('--path', type=dir_path, required=True,
    help='Path to result')
parser.add_argument('--samples', type=parse_int_or_iterable,
    help='Number of samples to test with (default: config.samples)')
parser.add_argument('--save', action='store_true',
    help='Save result to path')
args = parser.parse_args()

# Load model
config = read_config(args.path)
variables = restore_model(args.path)

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
if config.constrained:
    if config.ansatz in ['arnn-dense', 'arnn-conv1d', 'arnn-conv2d']:
        raise ValueError("ARNN Ansatze do not support constrained Hilbert spaces")
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)
else:
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# J1-J2 spin hamiltonian
J1 = config.J1
J2 = config.J2
msr_rot = -1.0 if config.msr else 1.0
sigmaz = [[1, 0], [0, -1]]
mszsz = (np.kron(sigmaz, sigmaz))
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
bond_operator = [
    (msr_rot*(J1/4)*mszsz).tolist(),
    ((J2/4)*mszsz).tolist(),
    (msr_rot*(J1/4)*exchange).tolist(),
    ((J2/4)*exchange).tolist(),
]
bond_color = [1, 2, 1, 2]
ha = nk.operator.GraphOperator(hilbert=hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)

def test(n_samples, config, variables, hilbert, operator):
    # Compute samples per rank
    if n_samples % n_nodes != 0:
        raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
    samples_per_rank = n_samples // n_nodes

    # Ansatz machine
    if config.dtype == 'real':
        dtype = jnp.float64
    elif config.dtype == 'complex':
        dtype = jnp.complex128
    if config.ansatz in ['qgps', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
        eps_init = gaussian(scale=config.scale, maxval=config.maxval, dtype=dtype)
    ma = FastARQGPSSymm(hilbert=hilbert, symmetries=g.automorphisms(), N=config.N, L=config.L**2, B=samples_per_rank,  eps_init=eps_init, dtype=dtype)

    # Sampler
    sa = ARDirectSampler(hilbert, n_chains_per_rank=samples_per_rank)

    # Variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, variables=variables)

    # Compute evergy
    energy = vs.expect(operator)

    return energy

# Parse sample sizes
if args.samples is None:
    args.samples = config.samples
if isinstance(args.samples, int):
    sample_sizes = [args.samples]
elif isinstance(args.samples, tuple):
    sample_sizes = args.samples

# Test energy expectation value
if rank == 0:
    output = {'energy': {}}
with tqdm(total=len(sample_sizes)) as pbar:
    it = 1
    for n_samples in sample_sizes:
        energy = test(n_samples, config, variables, hi, ha)
        pbar.set_postfix_str(f"n_samples = {n_samples}, energy = {energy.mean.real}")
        pbar.update(it)
        it += 1
        if args.save and rank == 0:
            output['energy'][n_samples] = float(energy.mean.real)

# Save
if args.save and rank == 0:
    with open(os.path.join(args.path, "test.log"), "w") as f:
        json.dump(output, f)
