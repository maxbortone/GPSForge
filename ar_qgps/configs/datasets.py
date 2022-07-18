from ml_collections import ConfigDict


def get_H2O_config():
    config = ConfigDict()
    config.basis = 'canonical'
    config.method = 'fci'
    config.select_largest = 100
    config.datapath = "/tmp/qGPSKet_data"
    return config