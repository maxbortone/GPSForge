import os
import yaml
import dataclasses


@dataclasses.dataclass
class Config:
    pass

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

def read_config(path : str, config_class=Config) -> Config:
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
    config = config_class(**config_copy)
    return config