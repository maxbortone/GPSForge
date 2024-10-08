from ml_collections import ConfigDict
from gps_forge.configs import common
from gps_forge.configs import systems
from gps_forge.configs import models
from gps_forge.configs import samplers
from gps_forge.configs.variational_states import get_MCState_config
from gps_forge.configs.optimizers import get_SRRMSProp_config


def get_config(modules) -> ConfigDict:
    # Base
    config = common.get_config()
    system, model, sampler = modules.split(',')

    # Training script
    config.trainer = 'vmc_block'

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
    config.variational_state_name = "MCState"
    config.variational_state = get_MCState_config(config)

    # Optimizer
    config.optimizer_name = "SRRMSProp"
    config.optimizer = get_SRRMSProp_config()
    config.optimizer.n_blocks = 2
    config.optimizer.n_samples_multiplier = 1

    return config.lock()