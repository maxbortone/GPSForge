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
    # Support dimension can be int or tuple
    M = config.model.M.split(',')
    if len(M) > 1:
        M = tuple(map(int, M))
    else:
        M = int(M[0])
    with config.ignore_type():
        config.model.M = M
    config = config.copy_and_resolve_references()
    return config