import os
import dataclasses
import configargparse
import jax; jax.config.update('jax_platform_name', 'cpu')
import netket as nk
import qGPSKet as qk
import numpy as np
import jax.numpy as jnp
from pyscf import scf, gto, ao2mo, fci
from utils import dir_path, save_config
from utils import MPIVars, compute_chunk_size
from utils import create_result
from utils import restore_model, select_checkpoint
from utils import Timer
from functools import partial


@dataclasses.dataclass
class Config:
    # System
    n_atoms : int = 10
    dist : float = 1.8

    # Ansatz
    M : int = 1
    sigma : float = 0.1

    # Sampler
    samples : int = 10

    # Optimizer
    iterations : int = 100
    learning_rate : float = 0.01
    diagonal_shift : float = 0.01
    sr_iterative : bool = True

def initialize_config(args):
    config = Config()
    config.n_atoms = args.n_atoms
    config.dist = args.dist
    config.M = args.M
    config.sigma = args.sigma
    config.samples = args.samples
    config.iterations = args.iterations
    config.learning_rate = args.learning_rate
    config.diagonal_shift = args.diagonal_shift
    config.sr_iterative = args.sr_iterative
    return config


def train():
    # Parser
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description='Train an Ansatz on a 1D hydrogen chain using VMC')
    parser.add_argument('-c', '--config', is_config_file=True,
        help='Path to configuration file')
    
    # System
    parser.add_argument('--n-atoms', type=int, default=10,
        help='Number of atoms (default: 10)')
    parser.add_argument('--dist', type=float, default=1.8,
        help='Distance between atoms in Bohr units (default: 1.8)')
    
    # Ansatz
    parser.add_argument('--M', type=int, default=1,
        help='Bond dimension of the QGPS Ansatz (default: 1)')
    parser.add_argument('--sigma', type=float, default=0.1,
        help='Standard deviation of the initial variational parameters (default: 0.1)')
    
    # Sampler
    parser.add_argument('--samples', type=int, default=1000,
        help='Number of samples used in VMC (default: 1000)')

    # Optimizer
    parser.add_argument('--iterations', type=int, default=100,
        help='Number of VMC iterations (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
        help='Learning rate of SGD (default: 0.01)')
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

    # Chunking
    parser.add_argument('--chunk-size-multiplier', type=float, default=1.0,
        help='Multiplier for the chunk size calculation (default: 1.0)')
    parser.add_argument('--set-chunk-size', action='store_true',
        help='Activate chunking on variational state, sets chunk_size=2**(ceil(log2(n_samples_per_rank*hilbert.size*multiplier)))')
    
    # Exact diagonalisation
    parser.add_argument('--compare-to-ed', action='store_true',
        help='Compare energy estimate to exact diagonalisation result')

    # Arguments
    args = parser.parse_args()

    # Config
    config = initialize_config(args)

    # Setup Hilbert space
    mol = gto.Mole()
    mol.build(
        atom = [('H', (x, 0., 0.)) for x in config.dist*np.arange(config.n_atoms)],
        basis = 'sto-6g',
        symmetry = True,
        unit="Bohr"
    )
    nelec = mol.nelectron
    print('Number of electrons: ', nelec)

    myhf = scf.RHF(mol)
    ehf = myhf.scf()
    norb = myhf.mo_coeff.shape[1]
    print('Number of molecular orbitals: ', norb)

    # Get hamiltonian elements
    # 1-electron 'core' hamiltonian terms, transformed into MO basis
    h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

    # Get 2-electron electron repulsion integrals, transformed into MO basis
    eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)

    # Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
    # Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
    h2 = ao2mo.restore(1, eri, norb)
    hi = qk.hilbert.FermionicDiscreteHilbert(N=norb, n_elec=(nelec//2,nelec//2))

    # Setup Hamiltonian
    ha = qk.operator.hamiltonian.AbInitioHamiltonianOnTheFly(hi, h1, h2, use_fast_update=False)

    # Setup Ansatz
    dtype = jnp.complex128
    init_fun = qk.nn.initializers.normal(sigma=config.sigma, dtype=dtype)
    count_spins = lambda spins: jnp.stack([jnp.zeros(spins.shape[0]), spins&1, (spins&2)/2, jnp.zeros(spins.shape[0])], axis=-1).astype(jnp.int32)
    def renormalize_log_psi(n_spins, hilbert, index):
        # 1. if the number of spin-up (spin-down) electrons until index
        #    is equal to n_elec_up (n_elec_down), then set to 0 the probability
        #    of sampling a singly occupied orbital with a spin-up (spin-down)
        #    electron, as well as the probability of sampling a doubly occupied orbital
        # 2. if the number of spin-up (spin-down) electrons that still need to be
        #    distributed, is smaller than the number of sites left, then set the probability
        #    of sampling an empty orbital to 0
        log_psi = jnp.zeros(hilbert.local_size)
        diff = jnp.array(hilbert._n_elec, jnp.int32)-n_spins[1:3]
        log_psi = jax.lax.cond(
            diff[0] == 0,
            lambda log_psi: log_psi.at[1].set(-jnp.inf),
            lambda log_psi: log_psi,
            log_psi
        )
        log_psi = jax.lax.cond(
            diff[1] == 0,
            lambda log_psi: log_psi.at[2].set(-jnp.inf),
            lambda log_psi: log_psi,
            log_psi
        )
        log_psi = jax.lax.cond(
            (diff == 0).any(),
            lambda log_psi: log_psi.at[3].set(-jnp.inf),
            lambda log_psi: log_psi,
            log_psi
        )
        log_psi = jax.lax.cond(
            (diff >= (hilbert.size-index)).any(),
            lambda log_psi: log_psi.at[0].set(-jnp.inf),
            lambda log_psi: log_psi,
            log_psi
        )
        return log_psi
    # count_spins = lambda spins: jnp.zeros((spins.shape[0], 4), jnp.int32)
    # renormalize_log_psi = lambda n_spins, hilbert, index: jnp.zeros((n_spins.shape[0], 4), jnp.complex128)
    ma = qk.models.ARqGPS(
        hi, config.M, dtype=dtype,
        init_fun=init_fun,
        count_spins=count_spins,
        renormalize_log_psi=jax.vmap(renormalize_log_psi, in_axes=(0, None, None))
    )

    # Compute samples per rank
    if config.samples % MPIVars.n_nodes != 0:
        raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
    samples_per_rank = config.samples // MPIVars.n_nodes

    # Sampler
    sa = qk.sampler.ARDirectSampler(hi, n_chains_per_rank=samples_per_rank, dtype=jnp.uint8)

    # Variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=config.samples)

    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
    sr = nk.optimizer.SR(qgt=partial(nk.optimizer.qgt.QGTJacobianDense, mode='complex', diag_shift=config.diagonal_shift), iterative=config.sr_iterative)

    # Load checkpoint
    if args.load_checkpoint_dir and os.path.isdir(args.load_checkpoint_dir):
        path = select_checkpoint(args.load_checkpoint_dir, args.load_checkpoint)
        variables = restore_model(path)
        vs.variables = variables

    # Set chunk size
    if args.set_chunk_size:
        vs.chunk_size = compute_chunk_size(args.chunk_size_multiplier, vs.n_samples_per_rank, ha.hilbert.size)

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
        # Get exact energy
        energy_mo, _ = fci.direct_spin1.FCI().kernel(h1, h2, norb, mol.nelectron)
        nuc_en = mol.energy_nuc()
        exact_energy = energy_mo + nuc_en

        # Get converged energy estimate
        data = logger.data
        var_energy = np.mean(data["Energy"]["Mean"][-10:].real)
        estimated_energy = var_energy + nuc_en
        if MPIVars.rank == 0:
            print(f"\nEstimated energy:\t{estimated_energy}")
            print(f"Exact energy:\t\t{exact_energy}")
            print(f"Relative error:\t\t{abs((estimated_energy-exact_energy)/exact_energy)}", flush=True)


if __name__ == '__main__':
    train()