import argparse
import netket as nk
import jax.numpy as jnp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

    # Heisenberg spin hamiltonian
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    # Compute samples per rank
    if args.samples % n_nodes != 0:
        raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
    samples_per_rank = samples // n_nodes

    # Ansatz machine
    dtype = jnp.complex128
    eps_init = gaussian(scale=0.001, maxval=0.1, dtype=dtype)
    if ansatz == 'qgps':
        ma = QGPS(N=N, eps_init=eps_init, dtype=dtype)
    elif ansatz == 'arqgps':
        ma = ARQGPS(hilbert=hi, N=N, L=L, eps_init=eps_init, dtype=dtype)
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
    if sa.is_exact:
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

    return np.array([runtime_s, runtime_i, runtime_e])

# Set style
sns.set_theme(style='dark')

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
N = 2
L = np.array([10, 20, 40, 80], dtype=np.int32)
ansatze = ['qgps', 'arqgps', 'arqgps-fast', 'arqgps-fast-symm']
modes = ['sampling', 'inference', 'evaluation']
data = []
for ansatz in ansatze:
    print(f"Running benchmarks for {ansatz}:")
    for l in tqdm(L):
        runtimes = benchmark(int(l), N, ansatz, args.samples)
        for mode, runtime in zip(modes, runtimes):
            row = {}
            row['ansatz'] = ansatz
            row['L'] = l
            row['runtime'] = runtime
            row['mode'] = mode
            data.append(row)
df = pd.DataFrame(data=data)

# Plot
f = sns.relplot(
    data=df, kind='line',
    x='L', y='runtime',
    hue='ansatz', col='mode', legend=False)
for ax in f.axes.ravel():
    ax.plot(L, 1e-4*L, 'k--')
    ax.plot(L, 1e-4*L**2, 'k--')
f.axes[0, 0].legend(
    loc="upper left",
    handles=f.axes[0,0].lines[-2:],
    labels=[r"$\mathcal{O}(L)$", r"$\mathcal{O}(L^2)$"])
f.set(xscale='log', yscale='log')
for ax in f.axes.ravel():
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_minor_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
f.set_axis_labels("L", "runtime (s)")
f.set_titles("{col_name}")
f.tight_layout(w_pad=0)
plt.show()
