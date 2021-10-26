import os
import re
import argparse
import uuid
import yaml
import json
import jax
import numpy as np
import pandas as pd
from flax.serialization import msgpack_restore
from timeit import default_timer as timer
from datetime import timedelta


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def parse_range(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))

def parse_int_or_iterable(string):
    vals = string.split(',')
    if len(vals)>1:
        l = tuple([int(v) for v in vals])
    else:
        l = int(vals[0])
    return l

def create_result(path):
    uid = uuid.uuid4().hex
    result_path = os.path.join(path, uid)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    return result_path

def save_config(config, path):
    config = vars(config)
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
    return True

def read_config(path):
    if not os.path.isdir(path):
        raise ValueError("The provided config path does not exist")
    config_path = os.path.join(path, "config.yaml")
    if not os.path.isfile(config_path):
        raise ValueError(f"No config file found at {path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)
    return config

def unpack_result(path):
    config = read_config(path)
    with open(os.path.join(path, "output.log"), "r") as f:
        result = json.load(f)
    iters = np.array(result['Energy']['iters'])
    if config.dtype == 'real':
        try:
            energy = np.array(result['Energy']['Mean'], dtype=np.float64)
        except:
            energy = np.array(result['Energy']['Mean']['real'], dtype=np.float64)
    elif config.dtype == 'complex':
        energy_re = np.array(result['Energy']['Mean']['real'], dtype=np.float64)
        energy_im = np.array(result['Energy']['Mean']['imag'], dtype=np.float64)
        energy = energy_re+1j*energy_im
    variance = np.array(result['Energy']['Variance'], dtype=np.float64)
    sigma = np.array(result['Energy']['Sigma'], dtype=np.float64)
    r_hat = np.array(result['Energy']['R_hat'], dtype=np.float64)
    tau_corr = np.array(result['Energy']['TauCorr'], dtype=np.float64)
    data = {
        'iters': iters,
        'energy': energy,
        'variance': variance,
        'sigma': sigma,
        'r_hat': r_hat,
        'tau_corr': tau_corr
    }
    return data

def restore_model(path):
    with open(os.path.join(path, "output.mpack"), 'rb') as b:
        variables = msgpack_restore(b.read())
    return variables

def list_results(paths):
    if isinstance(paths, str):
        results = [os.path.join(paths, result) for result in os.listdir(paths)]
    elif isinstance(paths, list):
        results = []
        for path in paths:
            for result in os.listdir(path):
                results.append(os.path.join(path, result))
    configs = []
    for result in results:
        config = vars(read_config(result))
        config['path'] = result
        config['uuid'] = os.path.basename(result)
        configs.append(config)
    df = pd.DataFrame(configs)
    df = df.sort_values(['L', 'N'])
    df = df.reset_index(drop=True)
    return df

def get_exact_energy(model, config):
    exact_energy = None
    base_path = os.path.dirname(os.path.abspath(__file__))
    if model == "heisenberg1d":
        path = os.path.join(base_path, 'result_DMRG_Heisenberg_1D.csv')
        df = pd.read_csv(path, dtype={'L': np.int64, 'E': np.float64})
        if (df['L']==config.L).any():
            exact_energy = 4*df.loc[df['L']==config.L]['E'].values[0]
    elif model == "j1j22d":
        path = os.path.join(base_path, 'result_ED_J1J2_2D.csv')
        df = pd.read_csv(path, dtype={'L': np.int64, 'J1': np.float32, 'J2': np.float32, 'E/L^2': np.float32, 'E': np.float32})
        if ((df['L']==config.L) & (df['J1']==config.J1) & (df['J2']==config.J2)).any():
            exact_energy = df.loc[(df['L']==config.L) & (df['J1']==config.J1) & (df['J2']==config.J2)]['E'].values[0]
    return exact_energy

def time_fn(fn, *args, repetitions=1, block=True):
    runtimes = []
    for _ in range(repetitions):
        start = timer()
        output = fn(*args)
        if block:
            jax.tree_map(lambda x: x.block_until_ready(), output)
        end = timer()
        runtime = timedelta(seconds=end-start)
        runtimes.append(runtime)
    return output, np.mean(runtimes)
