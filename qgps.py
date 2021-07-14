import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
from netket.utils.types import DType, Array, NNInitFunc
from initializers import gaussian


class QGPS(nn.Module):

    N: jnp.integer = 1
    dtype: DType = jnp.float64
    eps_init: NNInitFunc = gaussian()

    @nn.compact
    def __call__(self, inputs: Array):
        L = inputs.shape[-1]
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        eps = self.param("epsilon", self.eps_init, (2, self.N, L), self.dtype)

        def _compute_log_psi(eps, conf):
            idx = jnp.expand_dims(nn.relu(conf), axis=(0,1))
            vals = jnp.take_along_axis(eps, idx, axis=0)
            prods = jnp.prod(vals, axis=-1)
            log_psi = jnp.sum(prods)
            return log_psi
        
        log_psi = vmap(_compute_log_psi, in_axes=(None, 0))(eps, inputs)
        return log_psi
