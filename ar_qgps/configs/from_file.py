from fileinput import filename
import os
import glob
from VMCutils import read_config


def get_config(path):
    filenames = glob.glob(os.path.join(path, "*.yaml"))
    if len(filenames) > 0:
        _, filename = os.path.split(filenames[0])
        filename = filename.split('.')[0] 
        config = read_config(path, filename=filename)
    else:
        raise ValueError(f"No config found at {path}")
    return config