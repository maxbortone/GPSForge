import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import dir_path, read_config, unpack_result
from pprint import pprint


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
pprint(vars(config))

# Extract energies and errors
data = unpack_result(args.path)
energy = data['energy']
sigma = data['sigma']
iters = data['iters']

# Get exact energy
base_path = os.path.dirname(os.path.abspath(__file__))
if config.model == "heisenberg1d":
    path = os.path.join(base_path, 'result_DMRG_Heisenberg_1D.csv')
    df = pd.read_csv(path, dtype={'L': np.int64, 'E': np.float64})
    exact_energy = 4*df.loc[df['L']==config.L]['E'].values[0]
elif config.model == "j1j22d":
    path = os.path.join(base_path, 'result_ED_J1J2_2D.csv')
    df = pd.read_csv(path, dtype={'L': np.int64, 'J1': np.float32, 'J2': np.float32, 'E/L^2': np.float32, 'E': np.float32})
    exact_energy = 4*df.loc[(df['L']==config.L) & (df['J1']==config.J1) & (df['J2']==config.J2)]['E'].values[0]


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
if args.title:
    ax.set_title(args.title)
if args.save:
    plt.savefig(os.path.join(args.path, "energy.png"))

fig, ax = plt.subplots()
ax.plot(rel_error)
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Iteration')
ax.set_ylabel('Rel. error')
if args.title:
    ax.set_title(args.title)
if args.save:
    plt.savefig(os.path.join(args.path, "rel_error.png"))
plt.show()