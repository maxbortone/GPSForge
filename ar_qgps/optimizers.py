import jax
import netket as nk
import GPSKet as qk
from ml_collections import ConfigDict
from netket.vqs import VariationalState
from netket.optimizer import LinearOperator
from optax import GradientTransformation
from typing import Tuple, Optional, Union


_SOLVERS = {
    'pinv': qk.optimizer.pinv,
    'cg': jax.scipy.sparse.linalg.cg
}

def get_optimizer(config : ConfigDict, variational_state : Optional[VariationalState]=None) -> Tuple[GradientTransformation,Union[None, LinearOperator]]:
    """
    Return the optimizer specified in the config

    Args:
        config : experiment configuration file
        variational_state : variational state class used in the optimization (optional)

    Returns:
        the optimizer with preconditioner, if specified
    """
    if config.optimizer_name == 'Sgd' or config.optimizer_name == 'minSR':
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        sr = None
    elif config.optimizer_name == 'Adam':
        op = nk.optimizer.Adam(learning_rate=config.optimizer.learning_rate, b1=config.optimizer.b1, b2=config.optimizer.b2)
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
    elif config.optimizer_name == 'SRRMSProp':
        op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
        pars_struct = jax.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            variational_state.parameters
        )
        solver = _SOLVERS[config.optimizer.solver]
        sr = qk.optimizer.SRRMSProp(
            pars_struct,
            qk.optimizer.qgt.QGTJacobianDenseRMSProp,
            solver=solver,
            diag_shift=config.optimizer.diag_shift,
            decay=config.optimizer.decay,
            eps=config.optimizer.eps,
            mode=config.optimizer.mode
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} is not a valid class or is not supported yet.")
    return op, sr