import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import dir_path, read_config, unpack_result


# Parse arguments
parser = argparse.ArgumentParser(
    description='Plot result')
parser.add_argument('--path', type=dir_path, default='results',
    help='Path to result')
parser.add_argument('--title', type=str,
    help='Title of the plot')
parser.add_argument('--save', action='store_true',
    help='Save plots to path')
args = parser.parse_args()

# Set style
plt.style.use('seaborn-dark')

# Get config
config = read_config(args.path)

# Extract energies and errors
data = unpack_result(args.path)
energy = data['energy']
sigma = data['sigma']
iters = data['iters']

# Get exact energy
df = pd.read_csv('result_DMRG_Heisenberg_1D.csv', dtype={'L': np.int64, 'E': np.float64})
exact_energy = 4*df.loc[df['L']==config.L]['E'].values[0]

# Compute relative error
rel_error = np.abs((energy-exact_energy)/exact_energy)

# Plot
fig, ax = plt.subplots()
if config.dtype == 'complex':
    ax.plot(energy.real, label='real')
    ax.fill_between(iters, energy.real-sigma, energy.real+sigma, alpha=0.3)
    ax.plot(energy.imag, label='imag', alpha=0.3)
    ax.legend(loc='best')
else:
    ax.plot(energy)
    ax.fill_between(iters, energy-sigma, energy+sigma, alpha=0.3)
ax.axhline(exact_energy, linestyle='dashed', color='k')
ax.grid(True)
ax.set_xlabel('Iteration')
ax.set_ylabel('Energy')
if args.save:
    plt.savefig(os.path.join(args.path, "energy.png"))

fig, ax = plt.subplots()
ax.plot(rel_error)
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Iteration')
ax.set_ylabel('Rel. error')
if args.save:
    plt.savefig(os.path.join(args.path, "rel_error.png"))
plt.show()