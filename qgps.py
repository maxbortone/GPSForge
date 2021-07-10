import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
from netket.utils.types import DType, Array, NNInitFunc
from initializers import gaussian


def compute_log_psi(eps, conf):
    idx = jnp.expand_dims(nn.relu(conf), axis=(0,1))
    vals = jnp.take_along_axis(eps, idx, axis=0)
    prods = jnp.prod(vals, axis=-1)
    log_psi = jnp.sum(prods)
    return log_psi

class QGPS(nn.Module):

    N: jnp.integer = 1
    dtype: DType = jnp.float64
    eps_init: NNInitFunc = gaussian()

    @nn.compact
    def __call__(self, x_in: Array):
        L = x_in.shape[-1]
        if jnp.ndim(x_in) == 1:
            x_in = jnp.expand_dims(x_in, axis=0)
        eps = self.param("epsilon", self.eps_init, (2, self.N, L), self.dtype)
        log_psi = vmap(compute_log_psi, in_axes=(None, 0))(eps, x_in)
        return log_psi
