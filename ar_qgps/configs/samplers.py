from ml_collections import ConfigDict


def get_ARDirectSampler_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    return config

get_NKARDirectSampler_config = get_ARDirectSampler_config

def get_MetropolisExchange_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 1
    config.n_sweeps = parent.system.get_ref('Lx')
    config.d_max = parent.system.get_ref('Lx')//2
    return config

get_MetropolisFastExchange_config = get_MetropolisExchange_config

def get_MetropolisLocal_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 1
    config.n_sweeps = parent.system.get_ref('Lx')
    return config

def get_MetropolisHopping_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 1
    config.n_sweeps = 1
    config.hop_probability = 1.0
    return config

get_MetropolisFastHopping_config = get_MetropolisHopping_config