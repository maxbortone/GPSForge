import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from netket.utils.types import DType, Array, NNInitFunc, PyTree
from netket.models import ARNN
from initializers import gaussian


def l2_normalize(log_psi : Array) -> Array:
    return log_psi - 0.5*logsumexp(2*log_psi.real, axis=-1, keepdims=True)

def compute_conditional_log_psi(eps : Array, conf : Array) -> Array:
    idx = jnp.expand_dims(nn.relu(conf), axis=(0,1))
    vals = jnp.take_along_axis(eps, idx, axis=0)
    vals = jnp.pad(vals[:,:,:-1], ((0,0),(0,0),(1,0)), constant_values=1.0)
    vals = jnp.cumprod(vals, axis=-1)
    log_conds = jnp.sum(eps*vals, axis=1).T
    return log_conds # (L, 2)

class ARQGPS(ARNN):

    N: jnp.integer = 1
    L: jnp.integer = 1
    dtype: DType = jnp.float64
    eps_init: NNInitFunc = gaussian()

    def conditionals(self, x_in: Array, cache: PyTree):
        log_psi = vmap(compute_conditional_log_psi, in_axes=(None, 0))(self.eps, x_in)
        log_psi = l2_normalize(log_psi)
        p = jnp.exp(2*log_psi.real)
        return p, cache

    def setup(self):
        self.eps = self.param("epsilon", self.eps_init, (2, self.N, self.L), self.dtype)

    def __call__(self, x_in : Array):
        if jnp.ndim(x_in) == 1:
            x_in = jnp.expand_dims(x_in, axis=0)
        idx = nn.relu(x_in) # transforms [-1,+1]->[0,+1]
        idx = jnp.asarray(idx, dtype=jnp.int64)
        idx = jnp.expand_dims(idx, axis=-1)
        log_psi = vmap(compute_conditional_log_psi, in_axes=(None, 0))(self.eps, x_in)
        log_psi = l2_normalize(log_psi)
        log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)
        log_psi = jnp.sum(jnp.reshape(log_psi, (x_in.shape[0], -1)), axis=1)
        return log_psi