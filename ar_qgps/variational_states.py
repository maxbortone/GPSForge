import numpy as np
import netket as nk
import GPSKet as qk
from flax import linen as nn
from netket.hilbert import HomogeneousHilbert
from netket.sampler import Sampler
from ml_collections import ConfigDict
from typing import Optional
from VMCutils import MPIVars


_VARIATIONAL_STATES = {
    'ExactState': nk.vqs.ExactState,
    'MCState': nk.vqs.MCState,
    'MCStateUniqueSamples': qk.vqs.MCStateUniqueSamples,
    'MCStateStratifiedSampling': qk.vqs.MCStateStratifiedSampling
}

def get_variational_state(config : ConfigDict, model: nn.Module, hilbert : Optional[HomogeneousHilbert]=None, sampler : Optional[Sampler]=None) -> nk.vqs.VariationalState:
    """
    Return the variational state specified in the config

    Args:
        config : experiment configuration file
        model : model for the wavefunction Ansatz
        hilbert : Hilbert space on which the model should act (optional)
        sampler : sampler used to generate configurations (optional)

    Returns:
        the variational state
    """
    if config.variational_state_name != 'ExactState' and config.variational_state.seed is None:
        if MPIVars.rank == 0:
            seed = np.random.randint(np.iinfo(np.uint32).max)
        else:
            seed = None
        seed = MPIVars.comm.bcast(seed, root=0)
        config.variational_state.seed = seed
    try:
        vs_cls = _VARIATIONAL_STATES[config.get('variational_state_name')]
    except KeyError:
        raise ValueError(f"Variational state {config.variational_state_name} is not a valid class or is not supported yet.")
    args = []
    if 'MCState' in config.variational_state_name:
        if 'StratifiedSampling' in config.variational_state_name:
            sampler = qk.sampler.MetropolisHopping(hilbert, n_sweeps=config.variational_state.n_sweeps, n_chains_per_rank=1)
            if MPIVars.rank == 0:
                from ar_qgps.datasets import get_dataset

                dataset = get_dataset(config.system_name, config.variational_state.dataset)
                det_set_size = config.variational_state.deterministic_set_size
                norb = dataset[0].shape[1]
                det_set = np.zeros((det_set_size, norb), dtype=np.uint8)
                det_inds = np.argsort(np.abs(dataset[1]))[:-det_set_size-1:-1]
                np.copyto(det_set, dataset[0][det_inds,:])
                hilbert_size = dataset[0].shape[0]
            else:
                hilbert_size = None
                det_set = None
            hilbert_size = MPIVars.comm.bcast(hilbert_size, root=0)
            det_set = MPIVars.comm.bcast(det_set, root=0)
            args = [det_set, hilbert_size, sampler, model]
        else:
            args = [sampler, model]
    else:
        args = [hilbert, model]
    vs = vs_cls(*args, **config.variational_state)
    return vs