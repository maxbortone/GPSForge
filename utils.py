import os
import re
import argparse
import uuid
import yaml
import json
import numpy as np
from flax.serialization import msgpack_restore


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def parse_num_list(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))

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
        raise ValueError("The provided config path does not esist")
    with open(os.path.join(path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)
    return config

def unpack_result(path):
    config = read_config(path)
    with open(os.path.join(path, "output.log"), "r") as f:
        result = json.load(f)
    iters = np.array(result['Energy']['iters'])
    if config.dtype == 'real':
        energy = np.array(result['Energy']['Mean'], dtype=np.float64)
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
    with open(path, 'rb') as b:
        variables = msgpack_restore(b.read())
    return variables