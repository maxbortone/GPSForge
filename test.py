import os
import json
import numpy as np
from tqdm import tqdm
from utils import create_test_parser
from utils import select_checkpoint
from utils import read_config
from utils import MPIVars, setup_vmc
from utils import restore_model


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
        output = {}
    with tqdm(total=len(args.test_sample_sizes)) as pbar:
        it = 1
        for n_samples in args.test_sample_sizes:
            config.samples = n_samples
            ha, _, vs = setup_vmc(config)
            vs.variables = variables
            stats = vs.expect(ha)
            energy = stats.mean.item()
            pbar.set_postfix_str(f"n_samples = {n_samples}, energy = {energy.real}")
            pbar.update(it)
            it += 1
            if MPIVars.rank == 0:
                output[n_samples] = stats.to_dict()
                if np.iscomplex(energy):
                    output[n_samples]['Mean'] = {'real': energy.real, 'imag': energy.imag}

    # Save
    if args.save and MPIVars.rank == 0:
        with open(os.path.join(path, "test.json"), "w") as f:
            json.dump(output, f)


if __name__ == '__main__':
    test()