import os
import re
from typing import Union, Tuple


def parse_log_line(line : str) -> Tuple[int, Union[float, complex], float, float, float]:
    """Parses a line from the log file to extract optimization metrics"""
    chunks = line.split(' ')
    step = int(chunks[5].split('/')[0])
    mean = complex(chunks[8])
    sigma = float(chunks[10])
    variance = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", chunks[11])[0])
    r_hat = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", chunks[12])[0])
    return (step, mean, sigma, variance, r_hat)

def read_output_from_log(workdir: str, filename: str='train'):
    """Reads data from train logger file"""
    data = {'Energy':{'iters': [], 'Mean': {'real': [], 'imag': []}, 'Variance': [], 'Sigma': [], 'R_hat': []}}
    with open(os.path.join(workdir, f'{filename}.log'), 'r', encoding='UTF-8') as f:
        for line in f:
            if 'train.py:100' in line:
                step, mean, sigma, variance, r_hat = parse_log_line(line)
                data['Energy']['iters'].append(step)
                data['Energy']['Mean']['real'].append(mean.real)
                data['Energy']['Mean']['imag'].append(mean.imag)
                data['Energy']['Sigma'].append(sigma)
                data['Energy']['Variance'].append(variance)
                data['Energy']['R_hat'].append(r_hat)
    return data
