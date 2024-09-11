# GPSForge

GPSForge is a battle-hardened library for training wavefunction models based on the [Gaussian Process State](https://link.aps.org/doi/10.1103/PhysRevResearch.4.023126) with Variational Monte Carlo, with a particular focus on challenging quantum chemical systems.
All the models are implemented in [GPSKet](https://github.com/BoothGroup/GPSKet), a plugin for the [NetKet](https://github.com/netket/netket) framework, which uses [JAX](https://github.com/google/jax) to build machine-learning models of wavefunction Ansatze for quantum many-body problems.
The library takes a modular approach for quick and reproducible experimentation.

## Installation and Usage

To train and test the qGPS models working installations of [NetKet](https://github.com/netket/netket), the plugin [GPSKet](https://github.com/BoothGroup/GPSKet) and the library [ml_collections](https://github.com/google/ml_collections) for configuration files are required.
Once these are installed, clone or download this repository.

The main entrypoint to train and test the models on different systems is the `main.py` script.
To launch a new optimsation, if is sufficient to run the `main.py` script with the corresponding configuration file specifying the system, the Ansatz and the optimizer, as well as the working directory in which to save any output from the optimisation.
Importantly, configurations can be overwritten at the command line by specifying them as `--config.parameter=value`.

For example, to optimize the autoregressive qGPS on the 1D Heisenberg system with 10 sites, run
```
python -m gps_forge.main --config=$(pwd)/gps_forge/configs/vmc.py:Heisenberg1d,ARqGPS,ARDirectSampler,MCState,SRDense --workdir=$(pwd)/tmp/arqgps-$(date +%s) --config.total_steps=1000 --config.variational_state.n_samples=1000
```

The list of comma separated names after the path to the configuration file correspond to class names for the system, the Ansatz, the sampler, the variational state and the optimizer respectively.
Currently the following options are available:
    
- System:
    - `Heisenberg1d`: one-dimensional spin system with nearest neighbor Heisenberg interaction
    - `Heisenberg2d`: two-dimensional spin system with nearest neighbor Heisenberg interaction
    - `J1J22d`: two-dimensional spin system with nearest neighbor and next-nearest neighbor Heisenberg interaction
    - `Hchain`: one-dimensional chain of Hydrogen atoms separated by a certain interatomic distance
    - `Hring`: one-dimensional ring of Hydrogen atoms separated by a certain interatomic distance
    - `Hsheet`: two-dimensional sheet of Hydrogen atoms separated by a certain interatomic distance
    - `H2O`: water molecule
    - `N2`: nitrogen dimer
    - `Cr2`: Chromium dimer
    - `Hubbard1d`: one-dimensional fermionic Fermi-Hubbard system
    - `Hubbard2d`: two-dimensional fermionic Fermi-Hubbard system
- Ansatz:
    - `ARqGPSFull`: fully variational autoregressive qGPS model
    - `ARPlaquetteqGPS`: filter-based autoregressive qGPS model
    - `ARqGPS`: autoregressive qGPS model with weight-sharing
    - `qGPS`: qGPS model
    - `CPDBackflow`: backflow model with CP-decomposed orbitals
    - `SlaterqGPS`: Slater-Jastrow wavefunction with a qGPS Jastrow factor
- Sampler:
    - `ARDirectSampler`: direct sampler for autoregressive models
    - `MetropolisExchange`: Metropolis sampler with exchange rule for spin systems
    - `MetropolisHopping`: Metropolis sampler with hopping rule for fermionic systems
- Variational state:
    - `MCState`: Monte Carlo variational quantum state
    - `MCStateUniqueSamples`: Monte Carlo variational quantum state with support for unique samples
    - `MCStateStratifiedSampling`: Monte Carlo variational quantum state with support for stratified sampling
    - `FullSumState`: exact quantum state that computes expectation values over the whole Hilbert space
- Optimizer:
    - `Sgd`: Stochastic gradient descent optimizer
    - `Adam`: Adam optimizer
    - `SRDense`: Stochastic gradient descent optimizer with Stochastic Reconfiguration preconditioning of gradients (uses the dense Quantum Geometric Tensor)
    - `SRRMSProp`: SR with RMSProp diagonal shift
    - `kernelSR`: SR optimizer with a kernel trick for training large models

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

3. Install [JAX](https://github.com/google/jax):
```
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. Install [mpi4py](https://github.com/mpi4py/mpi4py):
```
pip install mpi4py
```

5. Install [mpi4jax](https://github.com/mpi4jax/mpi4jax):
```
pip install cython
pip install mpi4jax --no-build-isolation
```

6. Install [NetKet](https://github.com/netket/netket) with MPI support:
```
pip install "netket[mpi]"
```

7. Install [ml-collections](https://github.com/google/ml_collections):
```
pip install ml-collections
```

8. Create a bash script that binds GPU devices to individual MPI ranks:
```
cat << EOF > bind_gpu.sh
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

$@
EOF
chmod +x bind_gpu.sh
```

9. Run your job with as many ranks as available GPU devices (e.g. 4 ranks on a node with 4 GPUs):
```
mpirun -n 4 bind_gpu.sh python -m gps_forge.main --config=$(pwd)/gps_forge/configs/vmc.py:Heisenberg1d,ARqGPS,ARDirectSampler,MCState,SRDense --workdir=$(pwd)/tmp/arqgps-$(date +%s) --config.total_steps=1000 --config.variational_state.n_samples=1000
```
