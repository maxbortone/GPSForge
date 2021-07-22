import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from jax import lax
from typing import Any, List
from netket.utils.types import DType, Array, NNInitFunc


class FixedLayer(nn.Module):

    features: int
    kernel_init: NNInitFunc
    dtype: DType = jnp.float32
    precision: Any = None

    @nn.compact
    def __call__(self, inputs) -> Array:
        dtype = jnp.promote_types(inputs.dtype, self.dtype)
        inputs = jnp.asarray(inputs, dtype)
        in_features = inputs.shape[-1]
        shape = (in_features, self.features)
        kernel = self.variable("fixed", "kernel", self.kernel_init, shape, dtype)
        y = lax.dot_general(
            inputs,
            kernel.value,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            self.precision
        )
        return y

class VariationalLayer(nn.Module):

    bond_dim: int
    kernel_init: NNInitFunc
    dtype: DType = jnp.float32
    precision: Any = None

    @nn.compact
    def __call__(self, inputs) -> Array:
        dtype = jnp.promote_types(inputs.dtype, self.dtype)
        inputs = jnp.asarray(inputs, dtype)
        in_features = inputs.shape[-1]
        assert in_features % 2 == 0
        out_features = self.bond_dim*int(in_features/2)
        shape = (in_features, out_features)
        kernel = self.param("kernel", self.kernel_init, shape, dtype)
        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            self.precision
        )
        return y

BatchedFixedLayer = nn.vmap(
    FixedLayer,
    in_axes=1,
    out_axes=0,
    variable_axes={'fixed': None},
    split_rngs={'fixed': False}
)

BatchedVariationalLayer = nn.vmap(
    VariationalLayer,
    in_axes=0,
    out_axes=0,
    variable_axes={'params': 0},
    split_rngs={'params': True}
)
