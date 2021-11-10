import os
import json
import argparse
import jax.numpy as jnp
import netket as nk
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
    description='Test the energy of a wavefunction trained on the Heisenberg 1D model for different sample sizes')
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
    ma = FastARQGPSSymm(hilbert=hilbert, symmetries=g.automorphisms(), N=config.N, L=config.L, B=samples_per_rank,  eps_init=eps_init, dtype=dtype)

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
