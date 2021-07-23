from typing import Tuple
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from netket.hilbert import Spin
from netket.utils.types import DType, Array, NNInitFunc, PyTree
from netket.models import ARNN
from initializers import gaussian


def local_states_to_indices(hilbert : Spin, states : Array) -> Array:
    indices = (states+hilbert.local_size-1)/2
    indices = jnp.asarray(indices, jnp.int32)
    return indices

def l2_normalize(log_psi : Array) -> Array:
    return log_psi - 0.5*logsumexp(2*log_psi.real, axis=-1, keepdims=True)


class ARQGPS(ARNN):

    N: jnp.integer = 1
    L: jnp.integer = 1
    dtype: DType = jnp.float32
    eps_init: NNInitFunc = gaussian()

    def conditionals(self, inputs: Array, cache: PyTree) -> Tuple[Array, PyTree]:
        log_psi = _conditionals(self, inputs)
        p = jnp.exp(2*log_psi.real)
        return p, cache # (B, L, 2)

    def setup(self):
        self.eps = self.param("epsilon", self.eps_init, (2, self.N, self.L), self.dtype)

    def __call__(self, inputs: Array):
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        idx = local_states_to_indices(self.hilbert, inputs)
        idx = jnp.expand_dims(idx, axis=-1)
        log_psi = _conditionals(self, inputs)
        log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)
        log_psi = jnp.sum(jnp.reshape(log_psi, (inputs.shape[0], -1)), axis=1)
        return log_psi


def _conditionals(model, inputs):

    def _compute_conditional_log_psi(eps, input):
        idx = jnp.expand_dims(local_states_to_indices(model.hilbert, input), axis=(0,1))
        vals = jnp.take_along_axis(eps, idx, axis=0)
        vals = jax.lax.slice_in_dim(vals, 0, model.L-1, axis=2)
        vals = jnp.pad(vals, ((0,0),(0,0),(1,0)), constant_values=1.0)
        vals = jnp.cumprod(vals, axis=-1)
        log_conds = jnp.sum(jax.lax.mul(eps, vals), axis=1).T
        return log_conds # (L, 2)
    
    log_psi = vmap(_compute_conditional_log_psi, in_axes=(None, 0))(model.eps, inputs)
    log_psi = l2_normalize(log_psi)
    return log_psi # (B, L, 2)
