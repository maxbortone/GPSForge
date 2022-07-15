from ml_collections import ConfigDict


def get_ARDirectSampler_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    return config

def get_MetropolisExchange_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 4
    config.n_sweeps = parent.system.get_ref('Lx')*parent.system.get('Ly', 1)
    config.d_max = parent.system.get_ref('Lx')//2
    return config

def get_MetropolisLocal_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 4
    config.n_sweeps = parent.system.get_ref('Lx')*parent.system.get('Ly', 1)
    return config

def get_MetropolisHopping_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 4
    config.hop_probability = 1.0
    return config