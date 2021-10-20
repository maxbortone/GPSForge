import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from flax.core import freeze
from arqgps import ARQGPS, FastARQGPS


key_in, key_model = jax.random.split(jax.random.PRNGKey(np.random.randint(0, 100)))
L = 20
N = 2
B = 16
dtype = jnp.complex64
shape = (2, N, L)
g = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
inputs = hi.random_state(key_in, size=B)
# print(inputs)

# Initialize ARQGPS
arqgps = ARQGPS(hilbert=hi, N=N, L=L, dtype=dtype)
variables = arqgps.init(key_model, inputs)

# Initialize FastARQGPS
fast_arqgps = FastARQGPS(hilbert=hi, N=N, L=L, B=B, dtype=dtype)
variables = fast_arqgps.init(key_model, inputs, -1, method=fast_arqgps._conditional)

# Probabilities computed by `FastARQGPS.conditionals` and `ARQGPS.conditionals` should be the same
p1 = arqgps.apply(variables, inputs, method=arqgps.conditionals)
p2 = fast_arqgps.apply(variables, inputs, method=fast_arqgps.conditionals)

np.testing.assert_allclose(p2, p1)

# Cache of `FastARQGPS._conditional` at init should be equal to 1
cache = variables['cache']
ones = jnp.ones((B, 2, N))

np.testing.assert_allclose(cache['inputs'], ones)

# Iterating `FastARQGPS._conditional` over site indices
# should give same probabilities as those computed by `ARQGPS.conditionals`
p3 = jnp.zeros_like(p1)
params = variables['params']
for i in range(hi.size):
    variables = freeze({'params': params, 'cache': cache})
    p_i, mutables = fast_arqgps.apply(
        variables, inputs, i,
        method=fast_arqgps._conditional, mutable=['cache']
    )
    cache = mutables['cache']
    p3 = p3.at[:, i, :].set(p_i)

# Tolerance can't be too high since we are comparing values in exp space
np.testing.assert_allclose(p3, p1, rtol=1e-5)
