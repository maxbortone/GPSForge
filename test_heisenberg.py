import os
import json
import configargparse
import jax; jax.config.update('jax_platform_name', 'cpu')
import numpy as np
from utils import int_or_iterable, dir_path
from utils import select_checkpoint
from utils import read_config
from utils import MPIVars, setup_vmc, compute_chunk_size
from utils import restore_model
from utils import Timer
from .train_heisenberg import setup_vmc


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
    parser.add_argument('--save', action='store_true',
        help='Whether to save the test outcome or not; if True saves to same directory as checkpoint')

    # Chunking
    parser.add_argument('--chunk-size-multiplier', type=float, default=1.0,
        help='Multiplier for the chunk size calculation (default: 1.0)')
    parser.add_argument('--set-chunk-size', action='store_true',
        help='Activate chunking on variational state, sets chunk_size=2**(ceil(log2(n_samples_per_rank*hilbert.size*multiplier)))')

    return parser

def test():
    # Parser
    parser = create_test_parser('Test an optimized Ansatz on a Heisenberg system using VMC')
    args = parser.parse_args()

    # Config
    if args.load_checkpoint_dir and os.path.isdir(args.load_checkpoint_dir):
        path = select_checkpoint(args.load_checkpoint_dir, args.load_checkpoint)
    config = read_config(path)

    # Load checkpoint
    variables = restore_model(path)

    # Test energy evaluation
    if isinstance(args.sample_sizes, int):
        args.sample_sizes = [args.sample_sizes]
    if MPIVars.rank == 0:
        print(f"Running test for: \n{config}")
        print("Iteration\t Number of samples\t\t Energy statistics")
        output = {}
        n_sample_sizes = len(args.sample_sizes)
        t = Timer(n_sample_sizes)
    for step, n_samples in enumerate(args.sample_sizes):
        config.samples = n_samples
        ha, _, vs = setup_vmc(config)
        if args.set_chunk_size:
            vs.chunk_size = compute_chunk_size(args.chunk_size_multiplier, vs.n_samples_per_rank, ha.hilbert.size)
        vs.variables = variables
        stats = vs.expect(ha)
        if MPIVars.rank == 0:
            output[n_samples] = stats.to_dict()
            energy = stats.mean.item()
            if np.iscomplex(energy):
                output[n_samples]['Mean'] = {'real': energy.real, 'imag': energy.imag}
            t.update(step+1)
            print(f"[{step+1}/{n_sample_sizes}] n_samples: {vs.n_samples}, E: {stats} [{t.elapsed_time}<{t.remaining_time}, {t.runtime}s/it]")

    # Save
    if args.save and MPIVars.rank == 0:
        with open(os.path.join(path, "test.json"), "w") as f:
            json.dump(output, f)


if __name__ == '__main__':
    test()