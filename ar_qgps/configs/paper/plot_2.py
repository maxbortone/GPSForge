# Configuration file for Figure 2. of "Expressivity of autoregressive quantum states"
# The fully-variational autoregressive formulation above has cost $\mathcal{O}(L^2)$. This can be reduced by a factor of $\mathcal{O}(L)$ using one of these two strategies:
# 1. share weights between subsequent local correlators, i.e. parametrize them as $\psi_i = \psi_{\theta_{<i}}$
# 2. use the same translationally-invariant correlator for each site, i.e. $\psi_i = \psi_0$
import numpy as np
from ar_qgps.configs import vmc


def get_config(options):
    system, variant = options.split(",")
    if system not in ['Heisenberg2d', 'J1J22d']:
        raise ValueError(
            f"{system} is not a valid option. \
            Choose one of the following: \
            - 'Heisenberg2d' for a 2-D Heisenberg lattice \
            - 'J1J22d' for a 2-D Heisenberg lattice with frustration")
    if variant not in ['FV', 'WS', 'FI']:
        raise ValueError(
            f"{variant } is not a valid option. \
            Choose one of the following three: \
            - 'FV' for the fully-variational autoregressive GPS \
            - 'WS' for the autoregressive GPS with weight-sharing \
            - 'TI' for the autoregressive GPS with translationally-invariant correlator")
    if variant == "FV":
        ansatz = "ARqGPSFull"
    elif variant == "WS":
        ansatz = "ARqGPS"
    else:
        ansatz == "ARPlaquetteqGPS"

    modules = f"{system},{ansatz},ARDirectSampler,MCState,SgdSRDense"
    config = vmc.get_config(modules)

    # If the model needs to learn a sign structure and cannot output signed amplitudes,
    # i.e. the ground state is not positive and the model is an exponential,
    # then make sure to set the parameter dtype and the optimizer mode to complex
    if not config.system.sign_rule and config.model.get('apply_exp', True):
        config.model.dtype = "complex"
        config.optimizer.mode = "complex"

    config.variational_state.seed = np.random.randint(np.iinfo(np.uint32).max)
    config.variational_state.sampler_seed = np.random.randint(np.iinfo(np.uint32).max)

    return config.lock()

