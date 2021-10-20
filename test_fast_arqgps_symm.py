import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from matplotlib import pyplot as plt
from tqdm import tqdm
from flax.core.frozen_dict import freeze, unfreeze
from utils import restore_model
from jax.scipy.special import logsumexp
from arqgps import FastARQGPS, FastARQGPSSymm
from autoreg import ARDirectSampler
from initializers import gaussian


# Test #1
key = jax.random.PRNGKey(np.random.randint(0, 100))
L = 8
N = 2
B = 16
eps_init = gaussian(scale=0.01)

g = nk.graph.Chain(length=L, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
symmetries = g.automorphisms()
arqgps = FastARQGPS(hilbert=hi, N=N, L=L, B=B, eps_init=eps_init, dtype=jnp.complex64)
arqgps_symm = FastARQGPSSymm(hilbert=hi, symmetries=symmetries, N=N, L=L, B=B, eps_init=eps_init, dtype=jnp.complex64)
inputs = hi.random_state(key, B)
variables = arqgps_symm.init(key, inputs)

# Symmetrized amplitudes should be equal to average of
# amplitudes from non-symmetric model over
# symmetry transformed input configurations
T = symmetries.shape[0]
symmetries = symmetries.to_array()
log_psi_symm = arqgps_symm.apply(variables, inputs)
log_psi = jnp.zeros((T, B), dtype=jnp.complex64)
for t in tqdm(range(T)):
    inputs_t = jnp.take_along_axis(inputs, jnp.tile(symmetries[t], (B, 1)), 1)
    y = arqgps.apply(variables, inputs_t)
    log_psi = log_psi.at[t, :].set(y)
log_psi_real = 0.5*logsumexp(2*log_psi.real, axis=0, b=1/T)
log_psi_imag = logsumexp(1j*log_psi.imag, axis=0).imag
log_psi = log_psi_real+1j*log_psi_imag

np.testing.assert_allclose(log_psi_symm, log_psi)


# Test #2
# Load optimized Ansatz
# TODO: replace path with a test model
path = "./results/tests/fast_arqgps_symm_sampling/fe1185fc89e349c6b0f8b2886513685f"
params = restore_model(path)
variables = unfreeze(variables)
variables['params'] = params['params']
variables = freeze(variables)

# Frequency of sampled configurations should be proportional
# to squared amplitudes of symmetrized model
sa = ARDirectSampler(hi, n_chains_per_rank=B)
vs = nk.vqs.MCState(sa, arqgps_symm, n_samples=B, variables=variables)
S = 100000
M = S*B
confs = np.zeros(M)
confs_t = np.zeros(M)
T = symmetries.shape[0]
for i in tqdm(range(S)):
    samples = vs.sample()
    samples = jnp.squeeze(samples)
    confs[i*B:(i+1)*B] = hi.states_to_numbers(samples)
hist, edges = np.histogram(confs, bins=hi.n_states, range=(0, hi.n_states))
freqs = hist/M
inputs = hi.all_states()
A = len(inputs)//B
probs = np.zeros(len(inputs))
for i in tqdm(range(A)):
    log_psi = arqgps_symm.apply(variables, inputs[i*B:(i+1)*B, :])
    probs[i*B:(i+1)*B] = np.abs(np.exp(log_psi))**2

np.testing.assert_allclose(freqs, probs, atol=1e-3, rtol=1e-5)

fig, ax = plt.subplots()
ax.bar(edges[:-1]-0.25, freqs, width=0.5, color='C0', align='center', label="sampled")
ax.bar(np.arange(hi.n_states)+0.25, probs, width=0.5, color='C1', align='center', label="true")
ax.legend(loc="best")
ax.set_ylabel("Probability")
ax.set_xlabel("Configuration")
plt.tight_layout()
plt.show()
