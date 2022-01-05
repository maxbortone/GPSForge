import os
import numpy as np
import pandas as pd


def get_exact_energy(config):
    exact_energy = None
    base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    if config.Lx is not None and config.Ly is None:
        path = os.path.join(base_path, 'data/result_DMRG_Heisenberg_1D.csv')
        df = pd.read_csv(path, dtype={'L': np.int16, 'E': np.float64})
        if (df['L']==config.Lx).any():
            exact_energy = df.loc[df['L']==config.Lx]['E'].values[0]
    elif config.Lx is not None and config.Ly is not None:
        path = os.path.join(base_path, 'data/result_ED_J1J2_2D.csv')
        df = pd.read_csv(path, skiprows=0, header=1, dtype={'Lx': np.int16, 'Ly': np.int16, 'J1': np.float32, 'J2': np.float32, 'E/N': np.float32, 'E': np.float32})
        if ((df['Lx']==config.Lx) & (df['Ly']==config.Ly) & (df['J1']==config.J1) & (df['J2']==config.J2)).any():
            exact_energy = df.loc[(df['Lx']==config.Lx) & (df['Ly']==config.Ly) & (df['J1']==config.J1) & (df['J2']==config.J2)]['E'].values[0]
    return exact_energy

def get_literature_energy(model, ansatz, **filters):
    df = None
    base_path = os.path.dirname(os.path.abspath(__file__))
    if model == "heisenberg1d":
        pass
    elif model == "j1j22d":
        path = os.path.join(base_path, f"result_{ansatz.upper()}_J1J2_2D.csv")
        df = pd.read_csv(path, skiprows=0, header=1)
        # Filter rows
        for key, value in filters.items():
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                df = df.loc[df[key] == value]
            elif isinstance(value, list):
                df = df.loc[df[key].isin(value)]
            df['ansatz'] = ansatz
            df = df.reset_index(drop=True)
    return df