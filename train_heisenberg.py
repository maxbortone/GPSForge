import os
import pathlib
import configargparse
import dataclasses
import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np
import netket as nk
import qGPSKet as qk
import pandas as pd
from scipy.sparse.linalg import eigsh
from functools import partial
from VMCutils import bool_or_iterable, dir_path
from VMCutils import save_config
from VMCutils import MPIVars, compute_chunk_size
from VMCutils import create_result
from VMCutils import restore_model, select_checkpoint
from VMCutils import Timer


@dataclasses.dataclass
class Config:
    # System
    Lx : int = 6
    Ly : int = None
    J1 : float = 1.0
    J2 : float = 0.0
    constrained : bool = False
    sign_rule : bool = True

    # Ansatz
    ansatz : str = 'qgps'
    dtype : str = 'real'
    M : int = 1
    sigma : float = 0.1
    symmetries : str = 'all'

    # Sampler
    sampler : str = 'metropolis-exchange'
    samples : int = 1000
    discard : int = 100
    chains : int = 1
    sweeps : int = None

    # Optimizer
    iterations : int = 100
    optimizer : str = 'sgd-sr'
    learning_rate : float = 0.01
    b1 : float = 0.9
    b2 : float = 0.999
    diagonal_shift : float = 0.01
    sr_iterative : bool = True


def initialize_config(args : configargparse.ArgumentParser) -> Config:
    config = Config()

    # System
    config.Lx = args.Lx
    if args.Ly > 1:
        config.Ly = args.Ly
    else:
        config.Ly = None

    config.J1 = args.J1
    if args.J2 != 0.0:
        config.J2 = args.J2
    else:
        config.J2 = 0.0

    config.constrained = args.constrained
    config.sign_rule = args.sign_rule

    # Ansatz
    config.ansatz = args.ansatz
    config.M = args.M
    config.dtype = args.dtype
    config.sigma = args.sigma
    config.symmetries = args.symmetries

    # Sampler
    config.sampler = args.sampler
    config.samples = args.samples
    
    if args.sampler == 'metropolis-exchange':
        config.discard = args.discard
        config.chains = args.chains
        config.sweeps = args.sweeps
    else:
        config.discard = None
        config.chains = None

    # Optimizer
    config.iterations = args.iterations
    config.optimizer = args.optimizer
    config.learning_rate = args.learning_rate
    
    if args.optimizer == 'adam':
        config.b1 = args.b1
        config.b2 = args.b2
    else:
        config.b1 = None
        config.b2 = None
    
    if 'sr' in args.optimizer:
        config.diagonal_shift = args.diagonal_shift
        config.sr_iterative = args.sr_iterative
    else:
        config.diagonal_shift = None
        config.sr_iterative = None

    return config

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
    parser.add_argument('--sign-rule', type=bool_or_iterable, default=True,
        help='Whether to apply the Marshal Sign Rule or not (default: True)')
    
    # Ansatz
    parser.add_argument('--ansatz', default='qgps',
        choices=['qgps', 'arqgps', 'arqgps-full'],
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
    parser.add_argument('--sweeps', type=int, default=None,
        help='Number of sweeps used in MCMC sampler (default: None -> Hilbert size)')

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

    # Chunking
    parser.add_argument('--chunk-size', type=int,
        help='Chunk size used by the variational state to calculate expectation values')
    parser.add_argument('--chunk-size-multiplier', type=float, default=1.0,
        help='Multiplier for the chunk size calculation (default: 1.0)')
    parser.add_argument('--set-chunk-size', action='store_true',
        help='Activate chunking on variational state, sets chunk_size=2**(ceil(log2(n_samples_per_rank*hilbert.size*multiplier)))')
    
    # Exact diagonalisation
    parser.add_argument('--compare-to-ed', action='store_true',
        help='Compare energy estimate to exact diagonalisation result')

    return parser

def setup_vmc(config):
    # System
    if config.constrained:
        total_sz = 0.
    else:
        total_sz = None
    ha = qk.operator.hamiltonian.get_J1_J2_Hamiltonian(config.Lx, Ly=config.Ly, J1=config.J1, J2=config.J2, sign_rule=config.sign_rule, total_sz=total_sz)
    hi = ha.hilbert
    g = ha.graph

    # Ansatz model
    if config.dtype == 'real':
        dtype = jnp.float64
    elif config.dtype == 'complex':
        dtype = jnp.complex128
    init_fun = qk.nn.initializers.normal(sigma=config.sigma, dtype=dtype)
    to_indices = lambda x: jnp.asarray((x+hi.local_size-1)/2, jnp.int8)
    if isinstance(hi, nk.hilbert.Spin):
        if config.symmetries == 'all':
            automorphisms = True
            spin_flip = True
        elif config.symmetries == 'none':
            automorphisms = False
            spin_flip = False
        elif config.symmetries == 'automorphisms':
            automorphisms = True
            spin_flip = False
        elif config.symmetries == 'spin-flip':
            automorphisms = False
            spin_flip = True
    if config.ansatz == 'qgps':
        symmetries, inv_symmetries = qk.models.get_sym_transformation_spin(g, automorphisms=automorphisms, spin_flip=spin_flip)
        ma = qk.models.qGPS(
            hi, config.M,
            dtype=dtype, init_fun=init_fun,
            to_indices=to_indices, syms=(symmetries, inv_symmetries))
    elif 'arqgps' in config.ansatz:
        symmetries = g.automorphisms().to_array().T
        if automorphisms and spin_flip:
            def apply_symmetries(samples):
                out = jnp.take(samples, symmetries, axis=-1)
                out = jnp.concatenate((out, -out), axis=-1)
                return out
        elif automorphisms:
            def apply_symmetries(samples):
                out = jnp.take(samples, symmetries, axis=-1)
                return out
        elif spin_flip:
            def apply_symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                out = jnp.concatenate((out, -out), axis=-1)
                return out
        else:
            def apply_symmetries(samples):
                out = jnp.expand_dims(samples, axis=-1)
                return out
        if 'full' in config.ansatz:
            ma = qk.models.ARqGPSFull(
                hi, config.M,
                dtype=dtype, init_fun=init_fun,
                to_indices=to_indices,
                apply_symmetries=apply_symmetries)
        else:
            ma = qk.models.ARqGPS(
                hi, config.M,
                dtype=dtype, init_fun=init_fun,
                to_indices=to_indices,
                apply_symmetries=apply_symmetries)

    # Compute samples per rank
    if config.samples % MPIVars.n_nodes != 0:
        raise ValueError("Define a number of samples that is a multiple of the number of MPI ranks")
    samples_per_rank = config.samples // MPIVars.n_nodes

    # Sampler
    if config.sampler == 'ar-direct' and 'arqgps' in config.ansatz:
        sa = qk.sampler.ARDirectSampler(hi, n_chains_per_rank=samples_per_rank)
    elif config.sampler == 'metropolis-exchange':
        sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=config.chains, n_sweeps=config.sweeps, d_max=config.Lx//2)
    elif config.sampler == 'metropolis-local':
        sa = nk.sampler.MetropolisLocal(hi, n_chains_per_rank=config.chains, n_sweeps=config.sweeps)

    # Variational state
    if sa.is_exact:
        vs = nk.vqs.MCState(sa, ma, n_samples=config.samples)
    else:
        vs = nk.vqs.MCState(sa, ma, n_samples=config.samples, n_discard_per_chain=config.discard)

    # Optimizer
    if config.optimizer == 'sgd':
        op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
        sr = None
    elif config.optimizer == 'sgd-sr':
        op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
        sr = nk.optimizer.SR(diag_shift=config.diagonal_shift, iterative=config.sr_iterative)
    elif config.optimizer == 'sgd-sr-dense':
        op = nk.optimizer.Sgd(learning_rate=config.learning_rate)
        sr = nk.optimizer.SR(qgt=partial(nk.optimizer.qgt.QGTJacobianDense, mode=config.dtype, diag_shift=config.diagonal_shift), iterative=config.sr_iterative)
    elif config.optimizer == 'adam':
        op = nk.optimizer.Adam(learning_rate=config.learning_rate, b1=config.b1, b2=config.b2)
        sr = None

    return ha, (op, sr), vs

def get_exact_energy(hamiltonian, config):
    exact_energy = None
    base_path = pathlib.Path(__file__).parent.resolve()
    if config.Lx is not None and config.Ly is None:
        path = os.path.join(base_path, 'data/result_DMRG_Heisenberg_1D.csv')
        df = pd.read_csv(path, dtype={'L': np.int16, 'E': np.float64})
        if (df['L']==config.Lx).any():
            exact_energy = df.loc[df['L']==config.Lx]['E'].values[0]
    elif config.Lx is not None and config.Ly is not None:
        path = os.path.join(base_path, 'data/result_ED_J1J2_2D.csv')
        df = pd.read_csv(path, skiprows=0, header=1, dtype={'Lx': np.int16, 'Ly': np.int16, 'J1': np.float32, 'J2': np.float32, 'E/N': np.float32, 'E': np.float32})
        if ((df['Lx']==config.Lx) & (df['Ly']==config.Ly) & (df['J1']==config.J1) & (df['J2']==config.J2)).any():
            exact_energy = df.loc[(df['Lx']==config.Lx) & (df['Ly']==config.Ly) & (df['J1']==config.J1) & (df['J2']==config.J2)]['E'].values[0]
    if exact_energy is None:
        exact_energy = eigsh(hamiltonian.to_sparse(), k=1, which='SA', return_eigenvectors=False)[0]
    return exact_energy

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
        t.update(step+1)
        runtimes = MPIVars.comm.gather(t.runtime, root=0)
        if MPIVars.rank == 0:
            runtime = np.mean(runtimes)
        else:
            runtime = None
        runtime = MPIVars.comm.bcast(runtime, root=0)
        if MPIVars.rank == 0:
            grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
            grad_norm = np.linalg.norm(grad)
            print(f"[{step+1}/{config.iterations}] E: {vmc.energy}, ||∇E||: {grad_norm:.4f} [{t.elapsed_time}<{t.remaining_time}, {runtime:.4f}s/it]", flush=True)
        logger(step, {"Energy": vmc._loss_stats, "Runtime": runtime}, vmc.state)

    # Comparison with exact result
    if args.compare_to_ed:
        # Get converged energy estimate
        data = logger.data
        estimated_energy = np.mean(data["Energy"]["Mean"][-10:].real)

        # Get exact energy
        exact_energy = get_exact_energy(ha, config)
        if MPIVars.rank == 0:
            print(f"\nEstimated energy:\t{estimated_energy}")
            print(f"Exact energy:\t\t{exact_energy}")
            print(f"Relative error:\t\t{abs((estimated_energy-exact_energy)/exact_energy)}", flush=True)


if __name__ == '__main__':
    train()