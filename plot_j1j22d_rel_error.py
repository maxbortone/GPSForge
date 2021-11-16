import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_exact_energy, dir_path, get_literature_energy, list_results, read_config, unpack_result


# Parse arguments
parser = argparse.ArgumentParser(
    description='Plot relative error as a function of J2 and sample size')
parser.add_argument('--L', type=int, required=True,
    help='Number of sites in the system')
parser.add_argument('--paths', type=dir_path, nargs='+', required=True,
    help='Paths to results')
parser.add_argument('--y', type=str, default='rel_error', choices=['rel_error', 'energy_density'],
    help='Quantity to plot on the y-axis (default: rel_error)')
parser.add_argument('--compare', action='store_true', default=False,
    help='Whether to compare to literature values for similar system (default: False)')
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
df = df.reset_index()
df = df.sort_values(by=['ansatz', 'J2', 'samples'], ascending=[True, True, True])

# Extract energy estimates
num_rows = len(df.index)
energies = np.zeros(num_rows)
errors = np.zeros(num_rows)
energy_densities = np.zeros(num_rows)
rel_errors = np.zeros(num_rows)
for index, row in df.iterrows():
    config = read_config(row['path'])
    exact_energy = get_exact_energy('j1j22d', config)
    data = unpack_result(row['path'])
    energy = data['energy'][-1].real
    error = data['sigma'][-1]
    energy_density = np.abs(energy-exact_energy)/(args.L**2)
    rel_error = np.abs(energy-exact_energy)/np.abs(exact_energy) 
    energies[index] = energy
    errors[index] = error
    energy_densities[index] = energy_density
    rel_errors[index] = rel_error
df = df.assign(
    energy=pd.Series(energies), error=pd.Series(errors),
    energy_density=pd.Series(energy_densities), rel_error=pd.Series(rel_errors)
)

# Importe literature values
if args.compare:
    df_cnn = get_literature_energy('j1j22d', 'cnn', L=args.L)
    df_cnn['energy'] = args.L**2*df_cnn['E/L^2']
    num_rows = len(df_cnn.index)
    energy_densities = np.zeros(num_rows)
    rel_errors = np.zeros(num_rows)
    drop_rows = []
    for index, row in df_cnn.iterrows():
        config = argparse.Namespace(L=args.L, J1=row['J1'], J2=row['J2'])
        exact_energy = get_exact_energy('j1j22d', config)
        if exact_energy is None:
            drop_rows.append(index)
        else:
            energy = row['energy']
            energy_density = np.abs(energy-exact_energy)/(args.L**2)
            rel_error = np.abs(energy-exact_energy)/np.abs(exact_energy)
            energy_densities[index] = energy_density
            rel_errors[index] = rel_error
    df_cnn = df_cnn.drop(drop_rows)
    df_cnn = df_cnn.assign(
        energy_density=pd.Series(energy_densities), rel_error=pd.Series(rel_errors)
    )

    # Concatenate data frames
    df = pd.concat([df, df_cnn])
    df = df.reset_index(drop=True)

print(df[['N', 'ansatz', 'J2', 'samples', 'energy', 'error', 'energy_density', 'rel_error', 'uuid']].to_markdown(index=False, tablefmt='github'))

# Plot
f = sns.relplot(
    data=df, kind='line',
    x='J2', y=args.y,
    hue='samples', style='ansatz',
    markers=True, dashes=False,
    palette='tab10', markersize=10)
f.axes[0,0].axvspan(0.4, 0.6, facecolor='gray', alpha=0.3)
f.tight_layout()
if args.y == 'rel_error':
    f.axes[0,0].set_ylabel(r"$|E_{\theta}-E_{gs}|/|E_{gs}|$")
elif args.y == 'energy_density':
    f.axes[0,0].set_ylabel(r"$|E_{\theta}-E_{gs}|/N_{sites}$")
if args.title:
    f.set_title(args.title)
if args.save:
    plt.savefig(os.path.join(args.save, f"{args.y}_L{args.L}.pdf"), bbox_inches='tight')
plt.show()
