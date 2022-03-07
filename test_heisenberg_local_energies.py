import os
import json
import jax; jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import netket as nk
from utils import select_checkpoint
from utils import read_config
from utils import MPIVars, compute_chunk_size
from utils import restore_model
from utils import Timer
from .test_heisenberg import create_test_parser
from .train_heisenberg import setup_vmc


def test():
    # Parser
    parser = create_test_parser('Test local energies for an optimized Ansatz on a Heisenberg system using VMC')
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
        n_sample_sizes = len(args.sample_sizes)
        t = Timer(n_sample_sizes)
    for step, n_samples in enumerate(args.sample_sizes):
        config.samples = n_samples
        ha, _, vs = setup_vmc(config)
        vs.variables = variables
        local_energies = np.zeros(n_samples, dtype=vs.model.dtype)
        σ, kernel_args = nk.vqs.get_local_kernel_arguments(vs, ha)
        σ = σ.reshape((-1, σ.shape[-1]))
        def logpsi(w, σ):
            return vs._apply_fun({"params": w, **vs._model_state}, σ)
        if args.set_chunk_size:
            vs.chunk_size = compute_chunk_size(args.chunk_size_multiplier, vs.n_samples_per_rank, ha.hilbert.size)
            local_estimator_fun = nk.vqs.get_local_kernel(vs, ha, vs.chunk_size)
            local_energies = local_estimator_fun(logpsi, vs.parameters, σ, kernel_args, chunk_size=vs.chunk_size)
        else:
            local_estimator_fun = nk.vqs.get_local_kernel(vs, ha)
            local_energies = local_estimator_fun(logpsi, vs.parameters, σ, kernel_args)
        local_energies, token = nk.utils.mpi.mpi_allgather_jax(local_energies)
        σ, _ = nk.utils.mpi.mpi_allgather_jax(σ, token=token)
        if MPIVars.rank == 0:
            local_energies = np.array(local_energies.flatten())
            output = {}
            output['local_energies'] = {'real': local_energies.real.tolist(), 'imag': local_energies.imag.tolist()}
            σ = np.array(σ.reshape((-1, σ.shape[-1])))
            output['samples'] = σ.tolist()
            energy = np.mean(local_energies)
            output['mean'] = {'real': energy.real, 'imag': energy.imag}
            t.update(step+1)
            print(f"[{step+1}/{n_sample_sizes}] n_samples: {vs.n_samples}, E: {energy.real} [{t.elapsed_time}<{t.remaining_time}, {t.runtime}s/it]")

        # Save
        if args.save and MPIVars.rank == 0:
            with open(os.path.join(path, f"test_local_energies_samples_{n_samples}.json"), "w") as f:
                json.dump(output, f)


if __name__ == '__main__':
    test()