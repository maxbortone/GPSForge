import os
import re
import configargparse


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def int_or_iterable(string):
    vals = string.split(',')
    if len(vals)>1:
        l = tuple([int(v) for v in vals])
    else:
        l = int(vals[0])
    return l

def bool_or_iterable(string):
    vals = string.split(',')
    if len(vals)>1:
        l = tuple([v == 'true' for v in vals])
    else:
        l = bool(vals[0])
    return l

def range(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise configargparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))
