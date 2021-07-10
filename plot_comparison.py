import os
import glob
import json
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
pattern = f"{args.path}/heisenberg1d_L{args.L}_N[{args.N[0]}-{args.N[-1]}]_*_{args.dtype}.log"
paths = glob.glob(pattern)

# Extract energy estimates
estimates = {'qgps': {}, 'ar-qgps': {}}
for path in paths:
    N, ansatz = extract_info(path)
    energy, error = unpack_result(path)
    estimates[ansatz][N] = (energy, error)

# Sort estimates by N
estimates = {
                'qgps': dict(sorted(estimates['qgps'].items())),
                'ar-qgps': dict(sorted(estimates['ar-qgps'].items()))
            }
print(estimates)

# Get exact energy
df = pd.read_csv('result_DMRG_Heisenberg_1D.csv', dtype={'L': np.int64, 'E': np.float64})
exact_energy = 4*df.loc[df['L']==args.L]['E'].values[0]

# Plot
fig, ax = plt.subplots()
ax.errorbar(estimates['qgps'].keys(), [value[0] for value in estimates['qgps'].values()],
            yerr=[value[1] for value in estimates['qgps'].values()], label='QGPS')
ax.errorbar(estimates['ar-qgps'].keys(), [value[0] for value in estimates['ar-qgps'].values()],
            yerr=[value[1] for value in estimates['qgps'].values()], label='AR-QGPS')
ax.axhline(exact_energy, linestyle='dashed', color='k', label='Exact')
ax.grid(True)
ax.legend(loc='best')
ax.set_xlabel('N')
ax.set_ylabel('Energy')
if args.title:
    ax.set_title(args.title)
if args.save:
    plt.savefig(os.path.join(args.save, f"comparison_L{args.L}_N{args.N[0]}-{args.N[-1]}_{args.dtype}.png"))
plt.show()
