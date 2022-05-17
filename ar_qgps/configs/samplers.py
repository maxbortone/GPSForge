from ml_collections import ConfigDict


def get_ARDirectSampler_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    return config

def get_MetropolisExchange_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 4
    system_name = parent.get_ref('system_name')
    if system_name in ['Heisenberg2d', 'J1J22d']:
        n_sites = parent.get_ref('Lx')*parent.get_ref('Ly')
    elif system_name == 'Heisenberg1d':
        n_sites = parent.get_ref('Lx')
    config.n_sweeps = n_sites
    config.d_max = parent.get_ref('Lx')//2
    return config

def get_MetropolisLocal_config(parent : ConfigDict) -> ConfigDict:
    config = ConfigDict()
    config.n_chains_per_rank = 4
    system_name = parent.get_ref('system_name')
    if system_name in ['Heisenberg2d', 'J1J22d']:
        n_sites = parent.get_ref('Lx')*parent.get_ref('Ly')
    elif system_name == 'Heisenberg1d':
        n_sites = parent.get_ref('Lx')
    config.n_sweeps = n_sites
    return config