from ml_collections import ConfigDict


def get_Sgd_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    return config

def get_SgdSR_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.diag_shift = 0.001
    config.diag_scale = 0.01
    config.iterative = True
    return config

def get_SgdSRDense_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.qgt = 'QGTJacobianDense'
    config.mode = 'holomorphic'
    config.diag_shift = 0.001
    config.diag_scale = 0.01
    config.iterative = True
    return config

def get_SgdminSR_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.mode = 'holomorphic'
    config.rcond = 1e-12
    return config

def get_Adam_config() -> ConfigDict:
    config = ConfigDict()
    config.learning_rate = 0.01
    config.b1 = 0.9
    config.b2 = 0.999
    return config