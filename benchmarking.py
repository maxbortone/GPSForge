import argparse
import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from qgps import QGPS
from arqgps import ARQGPS, FastARQGPS, FastARQGPSSymm
from autoreg import ARDirectSampler
from initializers import gaussian
from utils import time_fn
from tqdm import tqdm


def benchmark(L, N, ansatz, samples):
    # 1D Lattice
    g = nk.graph.Chain(length=L, pbc=True)

    # Hilbert space of spins on the graph
    if ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
        hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    else:
        hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

    # Heisenberg spin hamiltonian
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    # Compute samples per rank
    if args.samples % n_nodes != 0:
        raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
    samples_per_rank = samples // n_nodes

    # Ansatz machine
    dtype = jnp.complex64
    eps_init = gaussian(scale=0.001, maxval=0.1, dtype=dtype)
    if ansatz == 'qgps':
        ma = QGPS(N=N, eps_init=eps_init, dtype=dtype)
    elif ansatz == 'arqgps':
        ma = ARQGPS(hilbert=hi, N=N, L=L,  eps_init=eps_init, dtype=dtype)
    elif ansatz == 'arqgps-fast':
        ma = FastARQGPS(hilbert=hi, N=N, L=L, B=samples_per_rank,  eps_init=eps_init, dtype=dtype)
    elif ansatz == 'arqgps-fast-symm':
        ma = FastARQGPSSymm(hilbert=hi, symmetries=g.automorphisms(), N=N, L=L, B=samples_per_rank,  eps_init=eps_init, dtype=dtype)

    # Sampler
    if ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
        sa = ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
    else:
        sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=1)

    # VMC state
    if ansatz in ['arqgps', 'arqgps-fast', 'arqgps-fast-symm']:
        vs = nk.vqs.MCState(sa, ma, n_samples=args.samples)
    else:
        vs = nk.vqs.MCState(sa, ma, n_samples=args.samples, n_discard_per_chain=int(args.samples/10))

    # Sampling
    _, _ = time_fn(vs.sample)
    samples, runtime_s = time_fn(vs.sample)

    # Inference
    samples = jnp.squeeze(samples)
    _, _ = time_fn(vs.log_value, samples)
    _, runtime_i = time_fn(vs.log_value, samples)

    # Evaluation
    samples = jnp.squeeze(samples)
    _, _ = time_fn(vs.expect, ha)
    _, runtime_e = time_fn(vs.expect, ha)

    return [runtime_s, runtime_i, runtime_e]

# Set style
plt.style.use('seaborn-dark')

# MPI variables
comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
rank = comm.Get_rank()
n_nodes = comm.Get_size()

# Parse arguments
parser = argparse.ArgumentParser(
    description='Benchmark Ansatze')
parser.add_argument('--samples', type=int, default=1000,
    help='Number of samples used in VMC (default: 1000)')
args = parser.parse_args()

# Run benchmark
N = 10
L = np.array([10, 20, 50, 100])
ansatze = ['qgps', 'arqgps-fast', 'arqgps-fast-symm']
runtimes = {
    'sampling': {ansatz: [] for ansatz in ansatze},
    'inference': {ansatz: [] for ansatz in ansatze},
    'evaluation': {ansatz: [] for ansatz in ansatze},
}
for ansatz in ansatze:
    print(f"Running benchmarks for {ansatz}:")
    for l in tqdm(L):
        runtime_s, runtime_i, runtime_e = benchmark(l, N, ansatz, args.samples)
        runtimes['sampling'][ansatz].append(runtime_s)
        runtimes['inference'][ansatz].append(runtime_i)
        runtimes['evaluation'][ansatz].append(runtime_e)
    runtimes['sampling'][ansatz] = np.array(runtimes['sampling'][ansatz])/runtimes['sampling'][ansatz][0]
    runtimes['inference'][ansatz] = np.array(runtimes['inference'][ansatz])/runtimes['inference'][ansatz][0]
    runtimes['evaluation'][ansatz] = np.array(runtimes['evaluation'][ansatz])/runtimes['evaluation'][ansatz][0]

# Plot
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(L, L/10, 'k--', label=r"$\mathcal{O}(L)$")
ax[0].plot(L, (L/10)**2, 'k-.', label=r"$\mathcal{O}(L^2)$")
ax[1].plot(L, L/10, 'k--')
ax[2].plot(L, (L/10)**2, 'k-.')
for i, ansatz in enumerate(ansatze):
    ax[0].plot(L, runtimes['sampling'][ansatz], color=f"C{i}", marker="o", linestyle="-", label=ansatz)
    ax[1].plot(L, runtimes['inference'][ansatz], color=f"C{i}", marker="o", linestyle="-")
    ax[2].plot(L, runtimes['evaluation'][ansatz], color=f"C{i}", marker="o", linestyle="-")
ax[0].set_yscale("log")
ax[2].set_yscale("log")
ax[2].set_xlabel("System size")
ax[0].set_ylabel("Runtime")
ax[1].set_ylabel("Runtime")
ax[2].set_ylabel("Runtime")
ax[0].set_title("Sampling")
ax[1].set_title("Inference")
ax[2].set_title("Evaluation")
ax[0].legend(loc="best")
ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
plt.show()
