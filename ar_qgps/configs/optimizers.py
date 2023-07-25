from ml_collections import ConfigDict


def get_Sgd_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    return config

def get_Adam_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.b1 = 0.9
    config.b2 = 0.999
    return config

def get_RMSProp_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.decay = 0.9
    config.eps = 1e-8
    return config

def get_SRDense_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.mode = 'real'
    config.diag_shift = 0.001
    config.diag_scale = 0.01
    config.solver = 'cg'
    return config

def get_SRRMSProp_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.mode = 'real'
    config.diag_shift = 0.01
    config.decay = 0.9
    config.eps = 1e-8
    config.solver = 'cg'
    return config

def get_minSR_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.mode = 'real'
    config.rcond = 1e-12
    config.diag_shift = 0.0
    return config