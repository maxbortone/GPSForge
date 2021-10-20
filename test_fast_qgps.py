import jax
import jax.numpy as jnp
import numpy as np
from qgps import QGPS, FastQGPS


key = jax.random.PRNGKey(np.random.randint(0, 100))
L = 4
N = 3
shape = (2, N, L)
inputs = jax.random.choice(key, jnp.array([-1., 1.]), (16, L))

# Test #1
# Run QGPS
qgps = QGPS(N, dtype=jnp.float64)
params = qgps.init(key, inputs)
log_psi_qgps = qgps.apply(params, inputs)

# Copy params
eps = params['params']['epsilon']
eps = jnp.transpose(eps, axes=(2,0,1))

# Run FastQGPS
fast_qgps = FastQGPS(N, dtype=jnp.float64)
params = fast_qgps.init(key, inputs)
params = params.copy({'params': {'VmapVariationalLayer_0': {'kernel': eps}}})
log_psi_fast_qgps = fast_qgps.apply(params, inputs)

np.testing.assert_allclose(log_psi_qgps, log_psi_fast_qgps)
