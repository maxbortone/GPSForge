import numpy as np
from VMCutils import MPIVars
from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Returns config values other than system, model, sampler, variational state and optimizer parameters."""

    config = ConfigDict()
    
    # Total number of training steps
    config.total_steps = 100
    # How often to report progress
    config.progress_every = 10
    # How often to write checkpoints
    config.checkpoint_every = 10
    # How often to log output to file
    config.log_every = 50

    # Will be set later
    # config.system_name = None
    # config.system = None
    # config.model_name = None
    # config.model = None
    # config.sampler_name = None
    # config.sampler = None
    # config.variational_state_name = None
    # config.variational_state = None
    # config.optimizer_name = None
    # config.optimizer = None

    return config

def resolve(config: ConfigDict) -> ConfigDict:
    # Set random seed
    if MPIVars.rank == 0:
        seed = np.random.randint(np.iinfo(np.uint32).max)
    else:
        seed = None
    seed = MPIVars.comm.bcast(seed, root=0)
    if config.get('variational_state', None) and config.variational_state_name != 'FullSumState' and config.variational_state.get('seed', None) is None:
        config.variational_state.seed = seed

    # Resolve molecular configuration
    if 'set_molecule' in config.system and callable(config.system.set_molecule):
        config = config.system.set_molecule(config)
        with config.ignore_type():
            # Replace the function with its name so we know how the molecule was set
            # This makes the ConfigDict object serialisable.
            if callable(config.system.set_molecule):
                config.system.set_molecule = config.system.set_molecule.__name__

    # Support dimension can be int or tuple
    if config.get('model', None) and config.model.get('M', None) and isinstance(config.model.M, str):
        M = config.model.M.split(',')
        if len(M) > 1:
            M = tuple(map(int, M))
        else:
            M = int(M[0])
        with config.ignore_type():
            config.model.M = M

    config = config.copy_and_resolve_references()
    return config.lock()