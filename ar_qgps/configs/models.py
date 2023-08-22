from ml_collections import ConfigDict


def get_qGPS_config() -> ConfigDict:
    config = ConfigDict()
    # Parse support dimension later, allow int as well as tuples
    config.M = '1'
    config.dtype = 'real'
    config.sigma = 0.1
    config.symmetries = 'none'
    config.apply_exp = True
    return config

get_PlaquetteqGPS_config = get_qGPS_config

def get_ARqGPS_config() -> ConfigDict:
    config = get_qGPS_config()
    config.normalize = True
    return config

get_ARqGPSFull_config = get_ARqGPS_config
get_ARPlaquetteqGPS_config = get_ARqGPS_config

def get_PixelCNN_config() -> ConfigDict:
    config = ConfigDict()
    config.dtype = 'real'
    config.kernel_size = 3
    config.n_channels = 32
    config.depth = 10
    config.normalize = True
    return config

def get_BackflowCPD_config() -> ConfigDict:
    config = ConfigDict()
    config.M = '1'
    config.dtype = 'real'
    config.init_fun = 'normal'
    config.sigma = 0.1
    config.restricted = False
    config.fixed_magnetization = True
    return config
