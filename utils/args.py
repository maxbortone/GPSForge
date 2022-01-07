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

def range(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise configargparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))

def create_parser(description):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description=description)
    parser.add_argument('-c', '--config', is_config_file=True,
        help='Path to configuration file')

    # System
    parser.add_argument('--Lx', '--L', type=int, default=6,
        help='Number of sites in first dimension of the system (default: 6)')
    parser.add_argument('--Ly', type=int, default=1,
        help='Number of sites in second dimension of the system (default: 1)')
    parser.add_argument('--J1', '--J', type=float, default=1.0,
        help='Nearest neighbor coupling (default: 1.0)')
    parser.add_argument('--J2', type=float, default=0.0,
        help='Next-nearest neighbor coupling (default: 0.0)')
    parser.add_argument('--constrained', type=bool, default=False,
        help='Whether to constrain the Hilbert space to the zero magnetization sector (default: False)')
    parser.add_argument('--sign-rule', type=bool, default=True,
        help='Whether to apply the Marshal Sign Rule or not (default: True)')
    
    # Ansatz
    parser.add_argument('--ansatz', default='qgps',
        choices=['qgps', 'arqgps'],
        help='Ansatz for the wavefunction (default: qgps)')
    parser.add_argument('--dtype', default='real',
        choices=['real', 'complex'],
        help='Type of the Ansatz parameters (default: real')
    parser.add_argument('--M', type=int, default=1,
        help='Bond dimension of the QGPS Ansatz (default: 1)')
    parser.add_argument('--sigma', type=float, default=0.1,
        help='Standard deviation of the initial variational parameters (default: 0.1)')
    parser.add_argument('--symmetries', default='all',
        choices=['all', 'none', 'automorphisms', 'spin-flip'],
        help='Symmetries of the Ansatz: \
             "all" for automorphisms and spin flip, \
             "none" for no symmetries, \
             "automorphisms" for only graph automorphisms, \
             "spin-flip" for only spin-flip (default: all)')
    
    # Sampler
    parser.add_argument('--sampler', default='metropolis-exchange',
        choices=['metropolis-exchange', 'ar-direct'],
        help='Sampler used in VMC (default: metropolis-exchange)')
    parser.add_argument('--samples', type=int, default=1000,
        help='Number of samples used in VMC (default: 1000)')
    parser.add_argument('--discard', type=int, default=100,
        help='Number of samples discarded during VMC (default: 100)')
    parser.add_argument('--chains', type=int, default=1,
        help='Number of chains used in VMC (default: 1)')

    # Optimizer
    parser.add_argument('--iterations', type=int, default=100,
        help='Number of VMC iterations (default: 100)')
    parser.add_argument('--optimizer', default='sgd-sr',
        choices=['sgd', 'sgd-sr', 'sgd-sr-dense', 'adam'],
        help='Optimizer used for learning (default: SGD)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
        help='Learning rate of SGD or Adam (default: 0.01)')
    parser.add_argument('--b1', type=float, default=0.9,
        help='Weight decay of 1st moment of Adam (default: 0.9)')
    parser.add_argument('--b2', type=float, default=0.999,
        help='Weight decay of 2nd moment of Adam (default: 0.999)')
    parser.add_argument('--diagonal-shift', type=float, default=0.01,
        help='Diagonal shift of SR (default: 0.01)')
    parser.add_argument('--sr-iterative', type=bool, default=True,
        help='Whether to use an iterative method or not for SR (default: True)')
    
    # Checkpointing
    parser.add_argument('--save-checkpoint-dir', type=dir_path,
        help='Directory to which checkpoints are saved')
    parser.add_argument('--checkpoint-every', type=int, default=50,
        help='Every how many interations should a chackpoint be flush to file (default: 50)')
    parser.add_argument('--load-checkpoint-dir', type=dir_path,
        help='Directory from which checkpoints are loaded')
    parser.add_argument('--load-checkpoint', type=str, default='best',
        help='Checkpoint or strategy used to select one from load_checkpoint_dir: \
             "{uuid}" pick checkpoint at load_checkpoint_dir/{uuid}, \
             "best" pick checkpoint with best energy, \
             "last" pick last checkpoint.')

    # Flags
    parser.add_argument('--set-chunk-size', action='store_true',
        help='Activate chunking on variational state, sets chunk_size=2**(ceil(log2(n_samples_per_rank*hilbert.size)))')
    parser.add_argument('--compare-to-ed', action='store_true',
        help='Compare energy estimate to exact diagonalisation result')

    return parser

def create_test_parser(description):
    parser = configargparse.ArgumentParser(description=description)

    # Testing
    parser.add_argument('--sample-sizes', type=int_or_iterable,
        help='Sample sizes on which to test the Ansatz')

    # Checkpointing
    parser.add_argument('--load-checkpoint-dir', type=dir_path,
        help='Directory from which checkpoints are loaded')
    parser.add_argument('--load-checkpoint', type=str, default='best',
        help='Checkpoint or strategy used to select one from load_checkpoint_dir: \
             "{uuid}" pick checkpoint at load_checkpoint_dir/{uuid}, \
             "best" pick checkpoint with best energy, \
             "last" pick last checkpoint.')

    # Flags
    parser.add_argument('--set-chunk-size', action='store_true',
        help='Activate chunking on variational state, sets chunk_size=2**(ceil(log2(n_samples_per_rank*hilbert.size)))')
    parser.add_argument('--save', action='store_true',
        help='Whether to save the test outcome or not; if True saves to same directory as checkpoint')

    return parser