import jax
import jax.numpy as jnp
import numpy as np
from layers import FixedLayer, VariationalLayer
from initializers import input_kernel_init, variational_kernel_init


key = jax.random.PRNGKey(np.random.randint(0, 100))
L = 4
N = 3
B = 16

# Test #1
inputs = jax.random.choice(key, jnp.array([-1.,1.]), (B, L))
inputs = jnp.asarray(inputs, jnp.float32)
layer = FixedLayer(
    features=int(2*L),
    kernel_init=input_kernel_init()
)
params = layer.init(key, inputs)
y_test = layer.apply(params, inputs)
kernel = np.array(params['fixed']['kernel']).T
y_true = np.zeros((16, 2*L))
for i in range(16):
    y_true[i] = np.dot(kernel, inputs[i])
np.testing.assert_allclose(y_test, y_true)

# Test #2
inputs = jax.random.choice(key, jnp.array([0.,1.]), (B, 2*L))
inputs = jnp.asarray(inputs, jnp.float32)
layer = VariationalLayer(
    bond_dim=N,
    kernel_init=variational_kernel_init()
)
params = layer.init(key, inputs)
y_test = layer.apply(params, inputs)
kernel = np.array(params['params']['kernel']).T
y_true = np.zeros((16, N*L))
for i in range(16):
    y_true[i] = np.dot(kernel, inputs[i])
np.testing.assert_allclose(y_test, y_true)
    