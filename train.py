import json
import os
import netket as nk
import numpy as np
from scipy.sparse.linalg import eigsh
from utils import create_parser
from utils import initialize_config, save_config
from utils import MPIVars, setup_vmc
from utils import create_result
from utils import get_exact_energy
from utils import restore_model, select_checkpoint


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

    # Variational Monte Carlo driver
    vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Run optimization
    if args.save_checkpoint_dir and os.path.isdir(args.save_checkpoint_dir):
        if MPIVars.rank == 0:
            path = create_result(args.save_checkpoint_dir)
            save_config(config, path)
            print(f"Checkpoint created at {path}")
        else:
            path = None
        path = MPIVars.comm.bcast(path, root=0)
        prefix = os.path.join(path, "output")
        logger = nk.logging.JsonLog(prefix)
        vmc.run(n_iter=config.iterations, out=logger, save_params_every=args.checkpoint_every, write_every=args.checkpoint_every)
    else:
        logger = nk.logging.RuntimeLog()
        vmc.run(n_iter=config.iterations, out=logger)

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
            print(f"Relative error:\t\t{abs((estimated_energy-exact_energy)/exact_energy)}")


if __name__ == '__main__':
    train()