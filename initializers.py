import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
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
            eps = jnp.asarray(eps, dtype)
        return eps
    return init

def input_kernel_init(dtype=np.float64):
    def init(shape, dtype=dtype):
        n_in, n_out = shape
        a = np.zeros(n_out)
        a[0] = 1.
        a[1] = -1.
        w = np.stack([np.roll(a, i*2) for i in range(n_in)])
        w = np.asarray(w, dtype=dtype)
        return w
    return init

def variational_kernel_init(loc=1.0, scale=1e-2, dtype=jnp.float64):
    def init(key, shape, dtype=dtype):
        n_in, n_out = shape
        N = int((n_out/n_in)*2)
        L = int(n_out/N)
        init_fn = gaussian(loc=loc, scale=scale, dtype=dtype)
        blocks = init_fn(key, (L, 2, N), dtype=dtype)
        w = block_diag(*blocks)
        w = jnp.asarray(w, dtype=dtype)
        return w
    return init
