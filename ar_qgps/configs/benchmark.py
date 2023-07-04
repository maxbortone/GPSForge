from ml_collections import ConfigDict
from ar_qgps.configs import systems
from ar_qgps.configs import models
from ar_qgps.configs import samplers
from ar_qgps.configs import variational_states
from ar_qgps.configs import optimizers


def get_config(modules) -> ConfigDict:
    # Base
    config = ConfigDict()
    config.repeat = 1
    system, model, sampler, variational_state, optimizer = modules.split(',')

    # Training script
    config.trainer = 'benchmark'

    # System
    get_system_config = getattr(systems, f"get_{system}_config")
    config.system_name = system
    config.system = get_system_config()

    # Model
    get_model_config = getattr(models, f"get_{model}_config")
    config.model_name = model
    config.model = get_model_config()

    # Sampler
    get_sampler_config = getattr(samplers, f"get_{sampler}_config")
    config.sampler_name = sampler
    config.sampler = get_sampler_config(config)   

    # Variational state
    get_variational_state_config = getattr(variational_states, f"get_{variational_state}_config")
    config.variational_state_name = variational_state
    config.variational_state = get_variational_state_config(config)

    # Optimizer
    get_optimizer_config = getattr(optimizers, f"get_{optimizer}_config")
    config.optimizer_name = optimizer
    config.optimizer = get_optimizer_config()

    return config.lock()