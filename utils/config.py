import os
from jax._src.api import checkpoint
import yaml
import configargparse
import dataclasses


@dataclasses.dataclass
class Config:
    # System
    Lx : int = 6
    Ly : int = None
    J1 : float = 1.0
    J2 : float = 0.0
    constrained : bool = False
    sign_rule : bool = True

    # Ansatz
    ansatz : str = 'qgps'
    dtype : str = 'real'
    M : int = 1
    sigma : float = 0.1
    symmetries : str = 'all'

    # Sampler
    sampler : str = 'metropolis-exchange'
    samples : int = 1000
    discard : int = 100
    chains : int = 1

    # Optimizer
    iterations : int = 100
    optimizer : str = 'sgd-sr'
    learning_rate : float = 0.01
    b1 : float = 0.9
    b2 : float = 0.999
    diagonal_shift : float = 0.01
    sr_iterative : bool = True


def initialize_config(args : configargparse.ArgumentParser) -> Config:
    config = Config()

    # System
    config.Lx = args.Lx
    if args.Ly > 1:
        config.Ly = args.Ly
    else:
        config.Ly = None

    config.J1 = args.J1
    if args.J2 != 0.0:
        config.J2 = args.J2
    else:
        config.J2 = 0.0

    config.constrained = args.constrained
    config.sign_rule = args.sign_rule

    # Ansatz
    config.ansatz = args.ansatz
    config.M = args.M
    config.dtype = args.dtype
    config.sigma = args.sigma
    config.symmetries = args.symmetries

    # Sampler
    config.sampler = args.sampler
    config.samples = args.samples
    
    if args.sampler == 'metropolis-exchange':
        config.discard = args.discard
        config.chains = args.chains
    else:
        config.discard = None
        config.chains = None

    # Optimizer
    config.iterations = args.iterations
    config.optimizer = args.optimizer
    config.learning_rate = args.learning_rate
    
    if args.optimizer == 'adam':
        config.b1 = args.b1
        config.b2 = args.b2
    else:
        config.b1 = None
        config.b2 = None
    
    if 'sr' in args.optimizer:
        config.diagonal_shift = args.diagonal_shift
        config.sr_iterative = args.sr_iterative
    else:
        config.diagonal_shift = None
        config.sr_iterative = None

    return config

def save_config(config : Config, path : str):
    config = dataclasses.asdict(config)
    config_copy = {}
    for key, value in config.items():
        if '_' in key:
            config_copy[key.replace('_', '-')] = value
        else:
            config_copy[key] = value
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.safe_dump(config_copy, f, default_flow_style=False, allow_unicode=True)

def read_config(path : str) -> Config:
    if not os.path.isdir(path):
        raise ValueError("The provided config path does not exist")
    config_path = os.path.join(path, "config.yaml")
    if not os.path.isfile(config_path):
        raise ValueError(f"No config file found at {path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_copy = {}
    for key, value in config.items():
        if '-' in key:
            config_copy[key.replace('-', '_')] = value
        else:
            config_copy[key] = value
    config = Config(**config_copy)
    return config