import os
import uuid
import json
import numpy as np
import pandas as pd
from flax.serialization import msgpack_restore
from dataclasses import asdict
from .config import Config, read_config


def create_result(path):
    name = uuid.uuid4().hex
    result_path = os.path.join(path, name)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    return result_path

def unpack_result(path, filename='output', config_class=Config):
    config = read_config(path, config_class=config_class)
    with open(os.path.join(path, f"{filename}.log"), "r") as f:
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

def unpack_test_result(path, filename="test"):
    with open(os.path.join(path, f"{filename}.json"), "r") as f:
        result = json.load(f)
    return result

def select_checkpoint(path, strategy_or_uuid):
    # TODO: implement strategies
    if strategy_or_uuid == "best":
        pass
    elif strategy_or_uuid == "last":
        pass
    elif isinstance(strategy_or_uuid, str):
        if os.path.isdir(os.path.join(path, strategy_or_uuid)):
            result = os.path.join(path, strategy_or_uuid)
        else:
            raise ValueError(f"Checkpoint {strategy_or_uuid} does not exist at {path}")
    else:
        raise ValueError("Provide a valid strategy or checkpoint id")
    return result

def restore_model(path, filename='output'):
    with open(os.path.join(path, f"{filename}.mpack"), 'rb') as b:
        variables = msgpack_restore(b.read())
    return variables

def list_results(paths, sort_by=None, config_class=Config):
    if isinstance(paths, str):
        results = [os.path.join(paths, result) for result in os.listdir(paths)]
    elif isinstance(paths, list):
        results = []
        for path in paths:
            for result in os.listdir(path):
                results.append(os.path.join(path, result))
    configs = []
    for result in results:
        config = asdict(read_config(result, config_class=config_class))
        config['path'] = result
        config['uuid'] = os.path.basename(result)
        configs.append(config)
    df = pd.DataFrame(configs)
    if sort_by:
        df = df.sort_values(sort_by)
    df = df.reset_index(drop=True)
    return df