from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from gps_forge.configs import common
from gps_forge.configs import systems
from gps_forge.configs import models
from gps_forge.configs import samplers
from gps_forge.configs import variational_states
from gps_forge.configs import optimizers


def get_config(modules) -> ConfigDict:
    # Base
    config = common.get_config()
    system, model, sampler, variational_state, optimizer = modules.split(',')

    # Training script
    config.trainer = 'vmc'

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

    # Descent finishing
    config.descent_finishing = ConfigDict()
    config.descent_finishing.total_steps = 50
    config.descent_finishing.learning_rate = 0.001

    # Evaluation configs
    config.evaluate = ConfigDict()
    config.evaluate.total_steps = 10
    config.evaluate.n_samples = 10*config.variational_state.get_ref('n_samples')
    config.evaluate.chunk_size = placeholder(int)

    return config.lock()