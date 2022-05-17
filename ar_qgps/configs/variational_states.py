from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_MCState_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_samples = 100
    if parent.get_ref('sampler_name') != 'ARDirectSampler':
        config.n_discard_per_chain = config.get_ref('n_samples') // 10
    config.chunk_size = placeholder(int)
    return config

def get_ExactState_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    return config