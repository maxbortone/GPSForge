# AR-qGPS

This repository holds the code to train and test the autoregressive form of the qGPS Ansatz, as well as its original formulation.
Both models are implemented in [qGPSKet](https://github.com/BoothGroup/qGPSKet), a plugin for the [NetKet](https://github.com/netket/netket) framework, which uses [JAX](https://github.com/google/jax) to build machine-learning models of wavefunction Ansatze for quantum many-body problems.

## Installation and Usage

To train and test the qGPS models working installations of [NetKet](https://github.com/netket/netket), the plugin [qGPSKet](https://github.com/BoothGroup/qGPSKet) and the library [ml_collections](https://github.com/google/ml_collections) for configuration files are required.
Once these are installed, clone or download this repository.

The main entrypoint to train and test the models on different systems is the `main.py` script.
To launch a new optimsation, if is sufficient to run the `main.py` script with the corresponding configuration file specifying the system, the Ansatz and the optimizer, as well as the working directory in which to save any output from the optimisation.
Importantly, configurations can be overwritten at the command line by specifying them as `--config.parameter=value`.

For example, to optimize the autoregressive qGPS on the 1D Heisenberg system with 10 sites, run
```
python -m ar_qgps.main --config=$(pwd)/ar_qgps/configs/qgps.py:Heisenberg1d,ARqGPS,ARDirectSampler,MCState,SgdSRDense --workdir=$(pwd)/tmp/arqgps-$(date +%s) --config.total_steps=1000 --config.variational_state.n_samples=1000
```

The list of comma separated names after the path to the configuration file correspond to class names for the system, the Ansatz, the sampler, the variational state and the optimizer respectively.
Currently the following options are available:
    
- System:
    - `Heisenberg1d`: one dimensional spin system with nearest neighbor Heisenberg interaction
    - `Heisenberg2d`: two dimensional spin system with nearest neighbor Heisenberg interaction
    - `J1J22d`: two dimensional spin system with nearest neighbor and next-nearest neighbor Heisenberg interaction
    - `Hchain`: one dimensional chain of Hydrogen atoms separated by a certain interatomic distance
    - `H2O`: water molecule
- Ansatz:
    - `ARqGPS`: autoregressive qGPS model with weight-sharing
    - `ARqGPSFull`: fully variational autoregressive qGPS model
    - `qGPS`: qGPS model
- Sampler:
    - `ARDirectSampler`: direct sampler for autoregressive models
    - `MetropolisExchange`: Metropolis sampler with exchange rule for spin systems
    - `MetropolisLocal`: Metropolis sampler with local rule for spin systems
- Variational state:
    - `MCState`: Monte Carlo variational quantum state
    - `ExactState`: exact quantum state (computes expectation values over the whole Hilbert space)
- Optimizer:
    - `Sgd`: Stochastic gradient descent optimizer
    - `SgdSR`: Stochastic gradient descent optimizer with Stochastic Reconfiguration preconditioning of gradients (uses the on-the-fly Quantum Geometric Tensor)
    - `SgdSRDense`: same as the above, but with the dense version of the QGT
    - `Adam`: Adam optimizer
