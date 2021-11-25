import abc
import jax
import jax.numpy as jnp
from typing import Union
from flax import linen as nn
from jax.scipy.special import logsumexp
from netket.hilbert import Spin
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils import HashableArray
from netket.utils.group import PermutationGroup
from netket.utils.types import DType, Array, NNInitFunc
from jax.nn.initializers import ones, zeros
from initializers import gaussian


def local_states_to_indices(hilbert : Spin, states : Array) -> Array:
    indices = (states+hilbert.local_size-1)/2
    indices = jnp.asarray(indices, jnp.int32)
    return indices

def normalize(log_psi : Array, machine_pow: int) -> Array:
    return log_psi - (1/machine_pow)*logsumexp(machine_pow*log_psi.real, axis=-1, keepdims=True)


class AbstractARQGPS(nn.Module):
    """
    Base class for autoregressive QGPS.

    Subclasses must implement the methods `__call__` and `conditionals`.
    They can also override `_conditional` to implement the caching for fast autoregressive sampling.

    They must also implement the field `machine_pow`,
    which specifies the exponent to normalize the outputs of `__call__`.
    """

    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous Hilbert spaces are supported."""

    # machine_pow: int = 2 Must be defined on subclasses

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

    def _conditional(self, inputs: Array, index: int) -> Array:
        """
        Computes the conditional probabilities for a site to take a given value.

        It should only be called successively with indices 0, 1, 2, ...,
        as in the autoregressive sampling procedure.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          index: index of the site.

        Returns:
          The probabilities with dimensions (batch, Hilbert.local_size).
        """
        return self.conditionals(inputs)[:, index, :]

    @abc.abstractmethod
    def conditionals(self, inputs: Array) -> Array:
        """
        Computes the conditional probabilities for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).

        Examples:

          >>> import pytest; pytest.skip("skip automated test of this docstring")
          >>>
          >>> p = model.apply(variables, σ, method=model.conditionals)
          >>> print(p[2, 3, :])
          [0.3 0.7]
          # For the 3rd spin of the 2nd sample in the batch,
          # it takes probability 0.3 to be spin down (local state index 0),
          # and probability 0.7 to be spin up (local state index 1).
        """

class ARQGPS(AbstractARQGPS):

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


class FastARQGPS(AbstractARQGPS):

    N: jnp.integer = 1
    L: jnp.integer = 1
    B: jnp.integer = 1
    dtype: DType = jnp.float64
    eps_init: NNInitFunc = gaussian(scale=0.0)
    machine_pow: int = 2

    def setup(self):
        self._eps = self.param("epsilon", self.eps_init, (2, self.N, self.L), self.dtype)
        self._cache = self.variable("cache", "inputs", ones, None, (self.B, 2, self.N), self.dtype)
        if self.hilbert.constrained:
            self._n_spins = self.variable("cache", "spins", zeros, None, (self.B, 2))

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


class FastARQGPSSymm(AbstractARQGPS):

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
        if self.hilbert.constrained:
            self._n_spins = self.variable("cache", "spins", zeros, None, (self.B, 2))
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


def _compute_conditional(hilbert, cache, n_spins, eps, inputs, index):
    # Slice inputs at index-1 to get cached products
    # (Note: when index=0, it doesn't matter what slice of the cache we take,
    # because it is initialized with ones)
    inputs_i = local_states_to_indices(hilbert, inputs[:, index-1]) # (B,)

    # Compute product of parameters and cache at index along bond dimension
    params_i = jnp.asarray(eps, eps.dtype)[:, :, index] # (2, N)
    prods = jax.vmap(lambda c, s: params_i*c[s], in_axes=(0, 0))(cache, inputs_i)
    prods = jnp.asarray(prods, eps.dtype) # (B, 2, N)

    # Update cache if index is positive, otherwise leave as is
    cache = jax.lax.cond(
        index >= 0,
        lambda _: prods,
        lambda _: cache,
        None
    )

    # Compute log conditional probabilities
    log_psi = jnp.sum(prods, axis=-1)

    # Update spins count if index is larger than 0, otherwise leave as is
    n_spins = jax.lax.cond(
        index > 0,
        lambda n_spins: n_spins + jnp.stack([jnp.abs(inputs_i-1), inputs_i], axis=-1),
        lambda n_spins: n_spins,
        n_spins
    )

    # If Hilbert space associated with the model is constrained, i.e.
    # model has "n_spins" in "cache" collection, then impose total magnetization.  
    # This is done by counting number of up/down spins until index, then if
    # n_spins is >= L/2 the probability of up/down spin at index should be 0,
    # i.e. the log probability becomes -inf
    # TODO: extend to cases where total_sz != 0
    log_psi = jax.lax.cond(
        index >= 0,
        lambda log_psi: log_psi+jnp.log(jnp.heaviside(hilbert.size//2-n_spins, 0)),
        lambda log_psi: log_psi,
        log_psi
    )

    return (cache, n_spins), log_psi

def _conditional(model, inputs, index):
    # Retrieve cache
    cache = model._cache.value
    cache = jnp.asarray(cache, model.dtype) # (B, 2, N)

    # Retrieve spins count
    n_spins = jnp.zeros((inputs.shape[0], 2))
    if model.has_variable("cache", "spins"):
        n_spins = model._n_spins.value
    
    # Compute log conditional probabilities
    (cache, n_spins), log_psi = _compute_conditional(model.hilbert, cache, n_spins, model._eps, inputs, index)
    log_psi = normalize(log_psi, model.machine_pow)
    
    # Update model cache
    if model.has_variable("cache", "inputs"):
        model._cache.value = cache
    if model.has_variable("cache", "spins"):
        model._n_spins.value = n_spins

    return log_psi # (B, 2)

def _conditionals(model, inputs):
    inputs = local_states_to_indices(model.hilbert, inputs)

    # Loop over sites while computing log conditional probabilities
    def _scan_fun(carry, index):
        cache, n_spins = carry
        (cache, n_spins), log_psi = _compute_conditional(model.hilbert, cache, n_spins, model._eps, inputs, index)
        n_spins = jax.lax.cond(
            model.hilbert.constrained,
            lambda _: n_spins,
            lambda _: jnp.zeros((inputs.shape[0], 2)),
            None
        )
        return (cache, n_spins), log_psi

    cache = jnp.ones((inputs.shape[0], 2, model.N), model.dtype)
    n_spins = jnp.zeros((inputs.shape[0], 2))
    indices = jnp.arange(model.hilbert.size)
    _, log_psi = jax.lax.scan(
        _scan_fun,
        (cache, n_spins),
        indices
    )
    log_psi = jnp.transpose(log_psi, [1, 0, 2])
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
