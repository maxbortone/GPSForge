import jax
import optax
import netket as nk
import GPSKet as qk
from ml_collections import ConfigDict
from netket.vqs import VariationalState
from netket.optimizer import LinearOperator
from optax import GradientTransformation, Schedule
from typing import Tuple, Optional, Union


_SOLVERS = {
    'pinv': qk.optimizer.pinv,
    'cg': jax.scipy.sparse.linalg.cg
}

def get_schedule(config : ConfigDict) -> Schedule:
    """
    Return the schedule specified in the config
    
    Args:
        config : configuration dictionary of the scheduled quantity

    Returns:
        optax schedule function
    """
    schedule_fn = getattr(optax, config.schedule_name)
    kwargs = config.to_dict()
    kwargs.pop('schedule_name')
    if 'boundaries_and_scales' in kwargs.keys():
        # NOTE: ml_collections doesn't allow int keys in dicts, this is a workaround
        boundaries_and_scales = kwargs.pop('boundaries_and_scales')
        boundaries = list(map(int, boundaries_and_scales.keys()))
        scales = list(map(float, boundaries_and_scales.values()))
        kwargs['boundaries_and_scales'] = dict(zip(boundaries, scales))
    return schedule_fn(**kwargs)

def get_optimizer(config : ConfigDict, variational_state : Optional[VariationalState]=None) -> Tuple[GradientTransformation,Union[None, LinearOperator]]:
    """
    Return the optimizer specified in the config

    Args:
        config : experiment configuration file
        variational_state : variational state class used in the optimization (optional)

    Returns:
        the optimizer with preconditioner, if specified
    """
    if config.optimizer_name in ['Sgd', 'minSR', 'kernelSR', 'SRRMSProp']:
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        sr = None
    elif config.optimizer_name == 'Adam':
        op = nk.optimizer.Adam(learning_rate=config.optimizer.learning_rate, b1=config.optimizer.b1, b2=config.optimizer.b2)
        sr = None
    elif config.optimizer_name == 'RMSProp':
        op = nk.optimizer.RmsProp(learning_rate=config.optimizer.learning_rate, beta=config.optimizer.decay, epscut=config.optimizer.eps)
        sr = None
    elif config.optimizer_name == 'SRDense':
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        qgt = nk.optimizer.qgt.QGTJacobianDense
        solver = _SOLVERS[config.optimizer.solver]
        sr = nk.optimizer.SR(
            qgt,
            solver=solver,
            diag_shift=config.optimizer.diag_shift,
            diag_scale=config.optimizer.diag_scale,
            mode=config.optimizer.mode)
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} is not a valid class or is not supported yet.")
    return op, sr