import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from jax.scipy.special import logsumexp
from arqgps import FastARQGPS, FastARQGPSSymm
from initializers import gaussian

key = jax.random.PRNGKey(2)
L = 8
N = 2
B = 2
eps_init = gaussian(scale=0.01)

g = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
symmetries = g.automorphisms()
arqgps = FastARQGPS(hilbert=hi, N=N, L=L, B=B, eps_init=eps_init)
arqgps_symm = FastARQGPSSymm(hilbert=hi, symmetries=symmetries, N=N, L=L, B=B, eps_init=eps_init)
inputs = hi.random_state(key, B)
variables = arqgps_symm.init(key, inputs)

# Symmetrized amplitudes should be equal to average of
# amplitudes from non-symmetric model over
# symmetry transformed input configurations
T = symmetries.shape[0]
symmetries = symmetries.to_array()
log_psi_symm = arqgps_symm.apply(variables, inputs)
log_psi = jnp.zeros((T, B))
for t in range(T):
    inputs_t = jnp.take_along_axis(inputs, jnp.tile(symmetries[t], (B, 1)), 1)
    y = arqgps.apply(variables, inputs_t)
    log_psi = log_psi.at[t, :].set(y)
log_psi_real = 0.5*logsumexp(2*log_psi.real, 0, 1/T)
log_psi_imag = logsumexp(1j*log_psi.imag, 0).imag
log_psi = log_psi_real+1j*log_psi_imag

np.testing.assert_allclose(log_psi_symm, log_psi)


