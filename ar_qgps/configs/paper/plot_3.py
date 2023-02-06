# Configuration file for Figure 3. of "Expressivity of autoregressive quantum states"
# Plot relative energy error as a function of the interaction strength U of a one-dimensional
# Hubbard system and for a filter autoregressive state of different support dimensions M
import numpy as np
from VMCutils import MPIVars
from ar_qgps.configs import vmc


def get_config(options):
    modules = f"Hubbard1d,ARPlaquetteqGPS,ARDirectSampler,MCState,SgdSRDense"
    config = vmc.get_config(modules)
    config.system.Lx = 32
    config.variational_state.n_samples = 4096
    config.optimizer.mode = config.model.dtype

    if MPIVars.rank == 0:
        seed = np.random.randint(np.iinfo(np.uint32).max)
    else:
        seed = None
    seed = MPIVars.comm.bcast(seed, root=0)
    config.variational_state.seed = seed

    return config.lock()