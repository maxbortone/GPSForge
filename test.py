import os
import json
import jax; jax.config.update('jax_platform_name', 'cpu')
import numpy as np
from utils import create_test_parser
from utils import select_checkpoint
from utils import read_config
from utils import MPIVars, setup_vmc, compute_chunk_size
from utils import restore_model
from utils import Timer


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