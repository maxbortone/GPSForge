import os
import numpy as np
import pandas as pd


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