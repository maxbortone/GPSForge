# AR-qGPS

This repository holds the code to train and test the autoregressive form of the qGPS Ansatz, as well as its original formulation.
Both models are implemented in [qGPSKet](https://github.com/BoothGroup/qGPSKet), a plugin for the [NetKet](https://github.com/netket/netket) framework, which uses [JAX](https://github.com/google/jax) to build machine-learning models of wavefunction Ansatze for quantum many-body problems.

## Installation and Usage

To train and test the qGPS models working installations of [NetKet](https://github.com/netket/netket) and the plugin [qGPSKet](https://github.com/BoothGroup/qGPSKet) are required.
Once these are installed, clone or download this repository.

To train a model run the `train.py` script with arguments specifying the system, the Ansatz and the optimizer (read the `help` message to see which arguments can be passed to it).

Likewise, to test a model, run `test.py` with the path to a previously trained model. The script will load the model and evaluate the variational energy of the Ansatz on the specified system for different sample sizes.
