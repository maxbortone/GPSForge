import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import parse_num_list, dir_path, read_config, unpack_result


# Parse arguments
parser = argparse.ArgumentParser(
    description='Plot comparison between different results')
parser.add_argument('--L', type=int, required=True,
    help='Number of sites in the system')
parser.add_argument('--N', type=parse_num_list, required=True,
    help='Range of bond dimensions')
parser.add_argument('--dtype', default='real', choices=['real', 'complex'],
    help='Type of the Ansatz parameters')
parser.add_argument('--paths', type=dir_path, nargs='+',
    help='Paths to results')
parser.add_argument('--title', type=str,
    help='Title of the plot')
parser.add_argument('--save', type=dir_path,
    help='Save plots to path')
args = parser.parse_args()

# Set style
plt.style.use('seaborn-dark')

def is_result_path(path):
    try:
        _ = read_config(path)
    except:
        return False
    else:
        return True

def parse_paths(paths):
    full_paths = set()
    for path in paths:
        if is_result_path(path):
            full_paths.add(path)
        else:
            paths += glob.glob(path + '/*')
    return full_paths

def select_results(full_paths, args):
    paths = []
    ansatze = set()
    for path in full_paths:
        config = read_config(path)
        if (config.L == args.L) and (config.N in args.N) and (config.dtype == args.dtype):
            paths.append(path)
            ansatz = f"{config.ansatz}-s{config.samples}"
            ansatze.add(ansatz)
    return paths, ansatze


# Select results
full_paths = parse_paths(args.paths)
result_paths, ansatze = select_results(full_paths, args)

# Extract energy estimates
estimates = {ansatz: {} for ansatz in ansatze}
for path in result_paths:
    config = read_config(path)
    data = unpack_result(path)
    energy = data['energy'][-1].real
    error = data['sigma'][-1]
    ansatz = f"{config.ansatz}-s{config.samples}"
    estimates[ansatz][config.N] = (energy, error)

# Sort ansatze alphabetically
ansatze = sorted(ansatze)

# Sort estimates by N
estimates = {ansatz: dict(sorted(runs.items())) for (ansatz, runs) in estimates.items()}
print(estimates)

# Get exact energy
df = pd.read_csv('result_DMRG_Heisenberg_1D.csv', dtype={'L': np.int64, 'E': np.float32})
exact_energy = 4*df.loc[df['L']==args.L]['E'].values[0]

# Plot
fig, ax = plt.subplots(2, 1, sharex=True)
for ansatz in ansatze:
    ns = np.array(list(estimates[ansatz].keys()))
    energies = [value[0] for value in estimates[ansatz].values()]
    errors = [value[1] for value in estimates[ansatz].values()]
    rel_errors = np.abs((energies-exact_energy)/exact_energy)/args.L
    ax[0].errorbar(ns, energies, yerr=errors, marker='o', label=ansatz.upper())
    ax[1].plot(ns, rel_errors, marker='o')
ax[0].axhline(exact_energy, linestyle='dashed', color='k', label='Exact')
ax[0].grid(True)
ax[0].legend(loc='best')
ax[0].set_ylabel('Energy')
ax[1].set_yscale('log')
ax[1].grid(True)
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel('N')
ax[1].set_ylabel('Rel. error (per site)')
if args.title:
    ax[0].set_title(args.title)
if args.save:
    plt.savefig(os.path.join(args.save, f"comparison_L{args.L}_N{args.N[0]}-{args.N[-1]}_{args.dtype}.png"))
plt.show()
