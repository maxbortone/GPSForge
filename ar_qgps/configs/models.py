from ml_collections import ConfigDict


def get_qGPS_config() -> ConfigDict:
    config = ConfigDict()
    config.M = 1
    config.dtype = 'complex'
    config.sigma = 0.1
    config.symmetries = 'all'
    return config

get_ARqGPS_config = get_qGPS_config
get_ARqGPSFull_config = get_qGPS_config