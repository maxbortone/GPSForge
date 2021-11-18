import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_exact_energy, parse_range, dir_path, list_results, read_config, unpack_result


# Parse arguments
parser = argparse.ArgumentParser(
    description='Compare different series of results as a function of bond dimension and sample size')
parser.add_argument('--L', type=int, required=True,
    help='Number of sites in the system')
parser.add_argument('--N', type=parse_range, required=True,
    help='Range of bond dimensions')
parser.add_argument('--paths', type=dir_path, nargs='+', required=True,
    help='Paths to results')
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
df = df.reset_index(drop=True)

# Get exact energy
config = read_config(df['path'][0])
exact_energy = get_exact_energy('heisenberg1d', config)

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

# Print table
df = df.sort_values(by=['N', 'ansatz', 'samples'], ascending=[True, True, True])
print(df[['N', 'ansatz', 'samples', 'energy', 'error', 'rel_error', 'uuid']].to_markdown(index=False, tablefmt='github'))

# Plot
f = sns.relplot(
    data=df, kind='line',
    x='N', y='rel_error',
    hue='samples', style='ansatz',
    markers=True, dashes=False,
    palette='tab10', markersize=10)
f.set(xscale='log', yscale='log')
f.tight_layout()
f.axes[0,0].set_ylabel(r"$|E_{\theta}-E_{gs}|/|E_{gs}|$")
f.axes[0,0].set_xticks(np.unique(df['N']))
f.axes[0,0].set_xticklabels(np.unique(df['N']))
# sns.move_legend(f, "upper right", bbox_to_anchor=(0.6, 0.9))
if args.title:
    f.set_title(args.title)
if args.save:
    plt.savefig(os.path.join(args.save, f"comparison_L{args.L}_N{args.N[0]}-{args.N[-1]}.png"))
plt.show()
