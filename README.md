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
python -m ar_qgps.main --config=$(pwd)/ar_qgps/configs/vmc.py:Heisenberg1d,ARqGPS,ARDirectSampler,MCState,SgdSRDense --workdir=$(pwd)/tmp/arqgps-$(date +%s) --config.total_steps=1000 --config.variational_state.n_samples=1000
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

## Setup on GPU nodes
To run code on GPU nodes in a HPC cluster, follow these steps.

1. First, log into a GPU node and load the necessary modules:
```
module load anaconda3/2021.05-gcc-9.4.0
module load cuda/11.4.2-gcc-9.4.0
module load cudnn/8.2.4.15-11.4-gcc-9.4.0
module load openmpi/4.1.1-gcc-9.4.0-ucx-python-3.8.12
```

2. Create a new conda environment:
```
conda create --name qgps-gpu python=3.8
conda activate qgps-gpu
```

3. Download and install the correct version of `jaxlib` (e.g. CUDA11, CuDNN 8.2 and Python 3.8):
```
pip install --upgrade pip
wget -v https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn82-cp38-none-manylinux2014_x86_64.whl
pip install jaxlib-0.3.8+cuda11.cudnn82-cp38-none-manylinux2014_x86_64.whl
```

4. Install [JAX](https://github.com/google/jax):
```
pip install "jax>=0.3.16,<0.4"
```

5. Install [mpi4py](https://github.com/mpi4py/mpi4py):
```
pip install mpi4py
```

6. Install [mpi4jax](https://github.com/mpi4jax/mpi4jax):
```
pip install cython
pip install mpi4jax --no-build-isolation
```

7. Install [NetKet](https://github.com/netket/netket) with MPI support:
```
pip install "netket[mpi]"
```

8. Install [ml-collections](https://github.com/google/ml_collections):
```
pip install ml-collections
```

9. Create a bash script that binds GPU devices to individual MPI ranks:
```
cat << EOF > bind_gpu.sh
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

$@
EOF
chmod +x bind_gpu.sh
```

10. Run your job with as many ranks as available GPU devices (e.g. 4 ranks on a node with 4 GPUs):
```
mpirun -n 4 bind_gpu.sh python -m ar_qgps.main --config=$(pwd)/ar_qgps/configs/vmc.py:Heisenberg1d,ARqGPS,ARDirectSampler,MCState,SgdSRDense --workdir=$(pwd)/tmp/arqgps-$(date +%s) --config.total_steps=1000 --config.variational_state.n_samples=1000
```
