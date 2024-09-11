from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_H2O_config():
    config = ConfigDict()
    config.basis = 'canonical'
    config.method = 'fci'
    config.select_largest = placeholder(int)
    config.datapath = "/tmp/GPSKet_data"
    return config