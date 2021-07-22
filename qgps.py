import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
from netket.utils.types import DType, Array, NNInitFunc
from initializers import gaussian, input_kernel_init, variational_kernel_init
from layers import BatchedFixedLayer, BatchedVariationalLayer


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

class FastQGPS(nn.Module):

    N: jnp.integer = 1
    dtype: DType = jnp.float32
    input_init_fn: NNInitFunc = input_kernel_init()
    variational_init_fn: NNInitFunc = variational_kernel_init()

    @nn.compact
    def __call__(self, inputs: Array):
        L = inputs.shape[-1]
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        
        # 1st layer maps input to indices
        inputs = jnp.expand_dims(inputs, axis=-1)
        y = BatchedFixedLayer(
            features=2,
            kernel_init=self.input_init_fn,
            dtype=self.dtype
        )(inputs)
        y = nn.relu(y)

        # 2nd layer takes values from variational tensor according to indices
        y = BatchedVariationalLayer(
            bond_dim=self.N,
            kernel_init=self.variational_init_fn,
            dtype=self.dtype
        )(y)
        y = jnp.log(y)

        # 3rd layer multiplies over L sites
        y = jnp.sum(y, axis=0)
        y = jnp.exp(y)

        # 4th layer sums over N bonds and outputs log-amplitude
        y = jnp.sum(y, axis=-1)

        return y
