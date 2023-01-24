from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_MCState_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_samples = 100
    if parent.get_ref('sampler_name') != 'ARDirectSampler':
        config.n_discard_per_chain = config.get_ref('n_samples') // 10
    config.chunk_size = placeholder(int)
    config.seed = placeholder(int)
    return config

def get_ExactState_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    return config

def get_MCStateUniqeSamples_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_samples = 100
    if parent.get_ref('sampler_name') != 'ARDirectSampler':
        config.n_discard_per_chain = config.get_ref('n_samples') // 10
    config.chunk_size = placeholder(int)
    config.max_sampling_steps = placeholder(int)
    config.fill_with_random = False
    return config

def get_MCStateStratifiedSampling_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_sweeps = 100
    config.n_samples = 100
    config.n_random_samples = placeholder(int)
    config.chunk_size = 1
    config.n_discard_per_chain = 100
    config.deterministic_set_size = 100
    from ar_qgps.configs import datasets
    get_dataset_config = getattr(datasets, f"get_{parent.system_name}_config")
    config.dataset = get_dataset_config()
    config.renormalize = False
    config.rand_norm = False
    return config