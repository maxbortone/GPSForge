# Configuration file for Figure 3. of "Expressivity of autoregressive quantum states"
# Plot relative energy error as a function of the interaction strength U of a one-dimensional
# Hubbard system and for a filter autoregressive state of different support dimensions M
from gps_forge.configs import vmc


def get_config():
    modules = f"Hubbard1d,ARPlaquetteqGPS,ARDirectSampler,MCState,SRDense"
    config = vmc.get_config(modules)
    config.system.Lx = 32
    config.variational_state.n_samples = 4096
    config.optimizer.mode = config.model.dtype

    return config.lock()