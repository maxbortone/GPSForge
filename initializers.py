import jax.numpy as jnp
from jax import random


def gaussian(loc=1.0, scale=1e-2, dtype=jnp.float64):
    def init(key, shape, dtype=dtype):
        if dtype == jnp.float64:
            eps = jnp.ones(shape)*loc+random.normal(key, shape, dtype)*scale
        else:
            key_mag, key_pha = random.split(key)
            mag = jnp.ones(shape, dtype=dtype)*loc+random.normal(key_mag, shape, dtype)*scale
            pha = random.uniform(key_pha, shape, minval=-0.1, maxval=0.1)
            eps = jnp.abs(mag)*jnp.exp(1j*pha)
        return eps
    return init