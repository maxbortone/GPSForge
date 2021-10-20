import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_exact_energy, parse_num_list, dir_path, list_results, read_config, unpack_result


# Parse arguments
parser = argparse.ArgumentParser(
    description='Compare different series of results for a model as a function of bond dimension and sample size')
parser.add_argument('--L', type=int, required=True,
    help='Number of sites in the system')
parser.add_argument('--N', type=parse_num_list, required=True,
    help='Range of bond dimensions')
parser.add_argument('--dtype', choices=['real', 'complex'],
    help='Type of the Ansatz parameters')
parser.add_argument('--paths', type=dir_path, nargs='+',
    help='Paths to results')
parser.add_argument('--model', default='heisenberg1d', choices=['heisenberg1d', 'j1j22d'],
    help='Model that has been simulated (default: heisenberg1d)')
parser.add_argument('--title', type=str,
    help='Title of the plot')
parser.add_argument('--save', type=dir_path,
    help='Save plots to path')
args = parser.parse_args()

# Set style
sns.set_theme(style='darkgrid')

# Select results
df = list_results(args.paths)
df = df.loc[df['L'] == args.L]
df = df.loc[df['N'].isin(args.N)]
if args.dtype:
    df = df.loc[df['dtype'] == args.dtype]
df = df.reset_index()
df = df.sort_values(by=['N', 'ansatz', 'dtype', 'msr', 'samples'], ascending=[True, True, False, True, True])

# Get exact energy
config = read_config(df['path'][0])
exact_energy = get_exact_energy(args.model, config)

# Extract energy estimates
num_rows = len(df.index)
energies = np.zeros(num_rows)
errors = np.zeros(num_rows)
rel_errors = np.zeros(num_rows)
for index, row in df.iterrows():
    data = unpack_result(row['path'])
    energy = data['energy'][-1].real
    error = data['sigma'][-1]
    rel_error = np.abs(energy-exact_energy)/np.abs(exact_energy)
    energies[index] = energy
    errors[index] = error
    rel_errors[index]= rel_error
df = df.assign(energy=pd.Series(energies), error=pd.Series(errors), rel_error=pd.Series(rel_errors))
print(df[['N', 'ansatz', 'dtype', 'msr', 'samples', 'energy', 'error', 'rel_error', 'uuid']].to_markdown(index=False, tablefmt='github'))

# Plot
g = sns.relplot(
    data=df, kind='line',
    x='N', y='rel_error',
    hue='ansatz', style='ansatz', size='samples',
    col='msr', row='dtype',
    markers=True, dashes=False)
g.set(yscale='log')
g.tight_layout()
if g.axes.shape == (2,2):
    g.fig.delaxes(g.axes[0,0])
if args.title:
    g.set_title(args.title)
if args.save:
    plt.savefig(os.path.join(args.save, f"comparison_L{args.L}_N{args.N[0]}-{args.N[-1]}.png"))
plt.show()
