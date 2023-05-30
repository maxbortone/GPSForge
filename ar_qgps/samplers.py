import numpy as np
import netket as nk
import GPSKet as qk
from netket.hilbert import HomogeneousHilbert
from netket.graph import AbstractGraph
from ml_collections import ConfigDict
from typing import Optional


_SAMPLERS = {
    'MetropolisExchange': nk.sampler.MetropolisExchange,
    'MetropolisFastExchange': qk.sampler.MetropolisFastExchange,
    'MetropolisHopping': qk.sampler.MetropolisHopping,
    'ARDirectSampler': qk.sampler.ARDirectSampler,
    'NKARDirectSampler': nk.sampler.ARDirectSampler
}

def get_sampler(config : ConfigDict, hilbert : HomogeneousHilbert, graph : Optional[AbstractGraph]=None) -> nk.sampler.Sampler:
    """
    Return the sampler specified in the config

    Args:
        config : experiment configuration file
        hilbert : Hilbert space on which the model should act
        graph : graph associated with the Hilbert space (optional)

    Returns:
        the sampler
    """
    if hasattr(config.model, 'normalize'):
        if not config.model.normalize and config.sampler_name == 'ARDirectSampler':
            raise ValueError("The ARDirectSampler can only be used with normalized autoregressive models")
    try:
        sa_cls = _SAMPLERS[config.get('sampler_name')]
    except KeyError:
        raise ValueError(f"Sampler {config.sampler_name} is not a valid class or is not supported yet.")
    kwargs = config.to_dict()['sampler']
    if config.system_name in ['Hchain', 'H2O'] and config.sampler_name != 'MetropolisHopping':
        kwargs['dtype'] = np.uint8
    if config.sampler_name == 'MetropolisExchange' or config.sampler_name == 'MetropolisFastExchange':
        kwargs['graph'] = graph
    sa = sa_cls(hilbert, **kwargs)
    return sa