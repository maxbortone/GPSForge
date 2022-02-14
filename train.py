import os
import jax; jax.config.update('jax_platform_name', 'cpu')
import netket as nk
import numpy as np
from scipy.sparse.linalg import eigsh
from utils import create_parser
from utils import initialize_config, save_config
from utils import MPIVars, setup_vmc, compute_chunk_size
from utils import create_result
from utils import get_exact_energy
from utils import restore_model, select_checkpoint
from utils import Timer


def train():
    # Parser
    parser = create_parser('Train an Ansatz on a Heisenberg system using VMC')
    args = parser.parse_args()

    # Config
    config = initialize_config(args)

    # Setup Hamiltonian, optimizer and variational state
    ha, (op, sr), vs = setup_vmc(config)

    # Load checkpoint
    if args.load_checkpoint_dir and os.path.isdir(args.load_checkpoint_dir):
        path = select_checkpoint(args.load_checkpoint_dir, args.load_checkpoint)
        variables = restore_model(path)
        vs.variables = variables

    # Set chunk size
    if args.chunk_size is None and args.set_chunk_size:
        vs.chunk_size = compute_chunk_size(args.chunk_size_multiplier, vs.n_samples_per_rank, ha.hilbert.size)
    elif args.chunk_size:
        vs.chunk_size = args.chunk_size

    # Variational Monte Carlo driver
    vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Logger
    if args.save_checkpoint_dir and os.path.isdir(args.save_checkpoint_dir):
        if MPIVars.rank == 0:
            path = create_result(args.save_checkpoint_dir)
            save_config(config, path)
            print(f"Checkpoint created at {path}")
        else:
            path = None
        path = MPIVars.comm.bcast(path, root=0)
        prefix = os.path.join(path, "output")
        logger = nk.logging.JsonLog(prefix, save_params_every=args.checkpoint_every, write_every=args.checkpoint_every)
    else:
        logger = nk.logging.RuntimeLog()

    # Run optimization
    if MPIVars.rank == 0:
        print(f"Running optimisation for: \n{config}")
        print("Iteration\t Energy statistics\t\t Gradient norm")
        t = Timer(config.iterations)
    for step in vmc.iter(config.iterations):
        logger(step, {"Energy": vmc._loss_stats}, vmc.state)
        if MPIVars.rank == 0:
            grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
            grad_norm = np.linalg.norm(grad)
            t.update(step+1)
            print(f"[{step+1}/{config.iterations}] E: {vmc.energy}, ||∇E||: {grad_norm} [{t.elapsed_time}<{t.remaining_time}, {t.runtime}s/it]", flush=True)

    # Comparison with exact result
    if args.compare_to_ed:
        # Get converged energy estimate
        data = logger.data
        estimated_energy = np.mean(data["Energy"]["Mean"][-10:].real)

        # Get exact energy
        exact_energy = get_exact_energy(config)
        if exact_energy is None:
            exact_energy = eigsh(ha.to_sparse(), k=1, which='SA', return_eigenvectors=False)[0]
        if MPIVars.rank == 0:
            print(f"\nEstimated energy:\t{estimated_energy}")
            print(f"Exact energy:\t\t{exact_energy}")
            print(f"Relative error:\t\t{abs((estimated_energy-exact_energy)/exact_energy)}", flush=True)


if __name__ == '__main__':
    train()