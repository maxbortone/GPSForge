# Configuration file for Figure 1. of "Expressivity of autoregressive quantum states"
# Going from a non-autoregressive product separable ansatz to an autoregressive one requires two steps:
# 1. introduce an autoregressive masking of the input
# 2. normalize the local correlators
# Here we start with a product separable ansatz and compare it to variants which are progressively more autoregressive, i.e. we compare the following three expressions:
# $$
# \begin{align}
#     \psi_{PS}(x) &= \prod\psi_i(x) \\
#     \psi_{MA}(x) &= \prod\psi_i(x_i;x_{<i}) \\
#     \psi_{AR}(x) &= \prod\psi_i(x_i;x_{<i})/Z_i,
# \end{align}
# $$
# where $Z_i = \sqrt{\sum_{x'}|\psi_i(x';x_{<i})|^2}$ is the normalization of the $i$-th local correlator.
import numpy as np
from VMCutils import MPIVars
from ar_qgps.configs import vmc


def get_config(options):
    system, variant = options.split(",")
    if system not in ['Heisenberg2d', 'J1J22d']:
        raise ValueError(
            f"{system} is not a valid option. \
            Choose one of the following: \
            - 'Heisenberg2d' for a 2-D Heisenberg lattice with Marshall Sign Rule (MSR) \
            - 'J1J22d' for a 2-D Heisenberg lattice with frustration")
    if variant not in ['PS', 'MA', 'AR']:
        raise ValueError(
            f"{variant } is not a valid option. \
            Choose one of the following three: \
            - 'PS' for the non-autoregressive product-separable GPS \
            - 'MA' for the masked GPS without normalization \
            - 'AR' for the autoregressive GPS")
    if variant == "PS":
        ansatz = "qGPS"
    else:
        ansatz = "ARqGPSFull"
    
    if variant == "AR":
        sampler = "ARDirectSampler"
    else:
        sampler = "MetropolisExchange"
    
    modules = f"{system},{ansatz},{sampler},MCState,SRRMSProp"
    config = vmc.get_config(modules)

    if variant == "MA":
        config.model.normalize = False
    elif variant == "AR":
        config.model.normalize = True

    # If the model needs to learn a sign structure and cannot output signed amplitudes,
    # i.e. the ground state is not positive and the model is an exponential,
    # then make sure to set the parameter dtype
    if not config.system.sign_rule and config.model.get('apply_exp', True):
        config.model.dtype = "complex"
    
    config.optimizer.mode = config.model.dtype

    return config.lock()

