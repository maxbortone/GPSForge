from ml_collections import ConfigDict
from ar_qgps.configs import common
from ar_qgps.configs import systems
from ar_qgps.configs import models
from ar_qgps.configs import samplers
from ar_qgps.configs.variational_states import get_MCState_config
from ar_qgps.configs.optimizers import get_SRDense_config


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
    config.optimizer_name = "BlockSR"
    config.optimizer = get_SRDense_config()
    config.optimizer.block_size = 2

    return config.lock()