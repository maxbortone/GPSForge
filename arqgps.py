from typing import Union
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from netket.hilbert import Spin
from netket.utils import HashableArray
from netket.utils.group import PermutationGroup
from netket.utils.types import DType, Array, NNInitFunc
from netket.models import AbstractARNN
from jax.nn.initializers import ones
from initializers import gaussian


def local_states_to_indices(hilbert : Spin, states : Array) -> Array:
    indices = (states+hilbert.local_size-1)/2
    indices = jnp.asarray(indices, jnp.int32)
    return indices

def normalize(log_psi : Array, machine_pow: int) -> Array:
    return log_psi - (1/machine_pow)*logsumexp(machine_pow*log_psi.real, axis=-1, keepdims=True)


class ARQGPS(AbstractARNN):

    N: jnp.integer = 1
    L: jnp.integer = 1
    dtype: DType = jnp.float64
    eps_init: NNInitFunc = gaussian(scale=0.0)
    machine_pow: int = 2

    def _conditional(self, inputs: Array, index: int) -> Array:
        log_psi = _conditionals(self, inputs)
        p = jnp.exp(self.machine_pow*log_psi.real)[:, index, :]
        return p # (B, 2)

    def conditionals(self, inputs: Array) -> Array:
        log_psi = _conditionals(self, inputs)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p # (B, L, 2)

    def setup(self):
        self._eps = self.param("epsilon", self.eps_init, (2, self.N, self.L), self.dtype)

    def __call__(self, inputs: Array):
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        return _call(self, inputs)


class FastARQGPS(AbstractARNN):

    N: jnp.integer = 1
    L: jnp.integer = 1
    B: jnp.integer = 1
    dtype: DType = jnp.float64
    eps_init: NNInitFunc = gaussian(scale=0.0)
    machine_pow: int = 2

    def setup(self):
        self._eps = self.param("epsilon", self.eps_init, (2, self.N, self.L), self.dtype)
        self._cache = self.variable("cache", "inputs", ones, None, (self.B, 2, self.N), self.dtype)

    def _conditional(self, inputs: Array, index: int) -> Array:
        log_psi = _conditional(self, inputs, index)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p # (B, 2)

    def conditionals(self, inputs: Array) -> Array:
        log_psi = _conditionals(self, inputs)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p # (B, L, 2)

    def __call__(self, inputs: Array) -> Array:
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        return _call(self, inputs)


class FastARQGPSSymm(AbstractARNN):

    symmetries: Union[HashableArray, PermutationGroup]
    N: jnp.integer = 1
    L: jnp.integer = 1
    B: jnp.integer = 1
    dtype: DType = jnp.float64
    eps_init: NNInitFunc = gaussian(scale=0.0)
    machine_pow: int = 2

    def setup(self):
        self._eps = self.param("epsilon", self.eps_init, (2, self.N, self.L), self.dtype)
        self._cache = self.variable("cache", "inputs", ones, None, (self.B, 2, self.N), self.dtype)
        self._symmetries = self.symmetries.to_array()
        self.n_symm = self.symmetries.shape[0]

    def _conditional(self, inputs: Array, index: int) -> Array:
        log_psi = _conditional(self, inputs, index)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p # (B, 2)

    def conditionals(self, inputs: Array) -> Array:
        log_psi = _conditionals(self, inputs)
        p = jnp.exp(self.machine_pow*log_psi.real)
        return p # (B, L, 2)

    def __call__(self, inputs: Array) -> Array:
        if jnp.ndim(inputs) == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        batch_size = inputs.shape[0]

        # Loop over symmetries and gather real and imaginary
        # part of average of the amplitudes separately
        def _scan_fun(carry, symmetry):
            (psi_symm_re, psi_symm_im) = carry
            transformed_inputs = jnp.take_along_axis(inputs, jnp.tile(symmetry, (batch_size, 1)), axis=1)
            log_psi = _call(self, transformed_inputs)
            psi_symm_re += jnp.exp(2*log_psi.real)
            psi_symm_im += jnp.exp(1j*log_psi.imag)
            return (psi_symm_re, psi_symm_im), None
        
        (psi_symm_re, psi_symm_im), _ = jax.lax.scan(
            _scan_fun,
            (jnp.zeros(batch_size, dtype=self.dtype), jnp.zeros(batch_size, dtype=self.dtype)),
            self._symmetries
        )

        # Compute symmetrized log-amplitudes
        log_psi_symm_re = 0.5*jnp.log(psi_symm_re/self.n_symm)
        log_psi_symm_im = jnp.log(psi_symm_im).imag
        log_psi_symm = log_psi_symm_re+1j*log_psi_symm_im
        return log_psi_symm # (B,)


def _conditional(model, inputs, index):
    # Slice inputs at index-1 to get cached products
    # (Note: when index=0, it doesn't matter what slice of the cache we take)
    inputs_i = local_states_to_indices(model.hilbert, inputs[:, index-1]) # (B,)

    # Retrieve cache
    cache = model._cache.value
    cache = jnp.asarray(cache, model.dtype) # (B, 2, N)
    
    # Compute product of parameters and cache at index along bond dimension
    params_i = jnp.asarray(model._eps[:, :, index], model.dtype)# (2, N)
    prods = jax.vmap(lambda c, s: params_i*c[s], in_axes=(0, 0))(cache, inputs_i) # (B, 2, N)
    prods = jnp.asarray(prods, model.dtype)

    # Update cache if index is positive, otherwise leave as is
    is_initialized = model.has_variable("cache", "inputs")
    if is_initialized:
        model._cache.value = jax.lax.cond(
            index >= 0,
            lambda _: prods,
            lambda _: cache,
            None
        )

    # Compute log-probabilities
    log_psi = jnp.sum(prods, axis=-1) # (B, 2)
    log_psi = normalize(log_psi, model.machine_pow)
    return log_psi # (B, 2)

def _compute_conditionals(eps, input):
    # Take values along spin-axis according to input
    idx = jnp.expand_dims(input, axis=(0, 1))
    vals = jnp.take_along_axis(eps, idx, axis=0)

    # Shift values to the right by 1 along site-axis,
    # fill first site with ones, and drop last one 
    vals = jnp.pad(vals[:, :, :-1], ((0, 0),(0, 0),(1, 0)), constant_values=1.0)

    # Compute cumulative product over sites
    vals = jnp.cumprod(vals, axis=-1)

    # Compute log-probabilities
    log_psi = jnp.sum(eps*vals, axis=1).T
    return log_psi # (L, 2)

def _conditionals(model, inputs):
    inputs = local_states_to_indices(model.hilbert, inputs)
    log_psi = jax.vmap(_compute_conditionals, in_axes=(None, 0))(model._eps, inputs)
    log_psi = normalize(log_psi, model.machine_pow)
    return log_psi # (B, L, 2)

def _call(model, inputs):
    batch_size = inputs.shape[0]

    # Compute log conditional probabilities
    log_psi = _conditionals(model, inputs)

    # Convert input configurations into indices
    idx = local_states_to_indices(model.hilbert, inputs)
    idx = jnp.expand_dims(idx, axis=-1)

    # Take log conditional probabilities along site-axis
    # according to input configurations
    log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)

    # Compute log-probability for each input configuration
    log_psi = jnp.sum(jnp.reshape(log_psi, (batch_size, -1)), axis=1)
    return log_psi # (B,)
