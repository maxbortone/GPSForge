import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import parse_num_list, dir_path, extract_info


# Parse arguments
parser = argparse.ArgumentParser(
    description='Plot comparison between QGPS and AR-QGPS')
parser.add_argument('-L', type=int, required=True,
    help='Number of sites in the system')
parser.add_argument('-N', type=parse_num_list, required=True,
    help='Range of bond dimensions')
parser.add_argument('--dtype', default='real', choices=['real', 'complex'],
    help='Type of the Ansatz parameters')
parser.add_argument('--path', type=dir_path, default='results',
    help='Path to results')
parser.add_argument('--title', type=str,
    help='Title of the plot')
parser.add_argument('--save', type=dir_path,
    help='Save plots to path')
args = parser.parse_args()

# Set style
plt.style.use('seaborn-dark')


def select_files(args):
    paths = []
    ansatze = set()
    for path in os.listdir(args.path):
        _, extension = os.path.splitext(path)
        if extension == ".log":
            N, ansatz = extract_info(path)
            if N in args.N:
                paths.append(os.path.join(args.path, path))
                ansatze.add(ansatz)
    return paths, ansatze

def unpack_result(path):
    with open(path, "r") as f:
        result = json.load(f)
    if args.dtype == 'real':
        energy = result['Energy']['Mean'][-1]
    elif args.dtype == 'complex':
        energy = result['Energy']['Mean']['real'][-1]
    error = result['Energy']['Sigma'][-1]
    if energy == None:
        energy = 0.0
    if error == None:
        error = 0.0
    return energy, error


# Select results
paths, ansatze = select_files(args)
print(paths)

# Extract energy estimates
estimates = {ansatz: {} for ansatz in ansatze}
for path in paths:
    N, ansatz = extract_info(path)
    energy, error = unpack_result(path)
    estimates[ansatz][N] = (energy, error)

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
