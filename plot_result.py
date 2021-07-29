import os
import glob
import json
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import dir_path, extract_info


# Parse arguments
parser = argparse.ArgumentParser(
    description='Plot result')
parser.add_argument('-L', type=int, required=True,
    help='Number of sites in the system')
parser.add_argument('-N', type=int, required=True,
    help='Bond dimensions')
parser.add_argument('--ansatz', default='qgps', choices=['qgps', 'ar-qgps', 'rbm', 'rbm-symm'],
    help='Ansatz for the wavefunction')
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


def unpack_result(path):
    with open(path, "r") as f:
        result = json.load(f)
    if args.dtype == 'real':
        energies = np.array(result['Energy']['Mean'], dtype=np.float64)
    elif args.dtype == 'complex':
        energies_re = np.array(result['Energy']['Mean']['real'], dtype=np.float64)
        energies_im = np.array(result['Energy']['Mean']['imag'], dtype=np.float64)
        energies = energies_re+1j*energies_im
    errors = np.array(result['Energy']['Sigma'], dtype=np.float64)
    energies = np.nan_to_num(energies)
    errors = np.nan_to_num(errors)
    return energies, errors

# Extract energies and errors
result_path = os.path.join(args.path, f"heisenberg1d_L{args.L}_N{args.N}_{args.ansatz}_{args.dtype}.log")
energies, errors = unpack_result(result_path)

# Get exact energy
df = pd.read_csv('result_DMRG_Heisenberg_1D.csv', dtype={'L': np.int64, 'E': np.float64})
exact_energy = 4*df.loc[df['L']==args.L]['E'].values[0]

# Compute relative error
rel_errors = np.abs((energies-exact_energy)/exact_energy)

# Plot
fig, ax = plt.subplots()
iterations = np.arange(len(energies))
if args.dtype == 'complex':
    ax.plot(energies.real, label='real')
    ax.fill_between(iterations, energies.real-errors, energies.real+errors, alpha=0.3)
    ax.plot(energies.imag, label='imag', alpha=0.3)
    ax.legend(loc='best')
else:
    ax.plot(energies)
    ax.fill_between(iterations, energies-errors, energies+errors, alpha=0.3)
ax.axhline(exact_energy, linestyle='dashed', color='k')
ax.grid(True)
ax.set_xlabel('Iteration')
ax.set_ylabel('Energy')
if args.save:
    plt.savefig(os.path.join(args.save, f"energy_heisenberg1d_L{args.L}_N{args.N}_{args.ansatz}_{args.dtype}"))

fig, ax = plt.subplots()
ax.plot(rel_errors)
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Iteration')
ax.set_ylabel('Rel. error')
if args.save:
    plt.savefig(os.path.join(args.save, f"error_heisenberg1d_L{args.L}_N{args.N}_{args.ansatz}_{args.dtype}"))
plt.show()