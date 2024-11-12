import sys
import os
import h5py
import math
from ml_collections.config_dict import placeholder
from ml_collections import ConfigDict


def closest_divisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a, n//a

def chain(config):
    config.system.molecule = [('H', (x*config.system.distance, 0., 0.)) for x in range(config.system.n_atoms)]
    return config

def ring(config):
    natoms = config.system.n_atoms
    dist = config.system.distance
    r = dist / (2 * math.sin(math.pi / natoms))
    molecule = []
    for i in range(natoms):
        theta = i * (2 * math.pi / natoms)
        molecule.append(('H', (r * math.cos(theta), r * math.sin(theta), 0.)))
    config.system.molecule = molecule
    return config

def sheet(config):
    Lx, Ly = closest_divisors(config.system.n_atoms)
    molecule = []
    dist = config.system.distance
    for x in range(Lx):
        for y in range(Ly):
            molecule.append(('H', (x*dist, y*dist, 0.)))
    config.system.molecule = molecule
    return config

def diatomic(config):
    atom1 = atom2 = config.system.molecule_name.strip('2')
    atoms = (atom1, atom2)
    pos = (config.system.bond_length * config.system.bond_length_multiple) / 2
    coords = ((-pos, 0., 0.), (pos, 0., 0.))
    config.system.molecule = [
        (a, c) for a, c in zip(atoms, coords)
    ]
    return config

def small_molecule(config):
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'small-molecules')
    molecule_name = config.system.molecule_name
    with h5py.File(os.path.join(dataset_path, f"{molecule_name}.hdf5"), 'r') as f:
        atoms = map(lambda x: x.decode('utf-8'), f['geometry']['atoms'][()])
        positions = map(lambda pos: tuple(float(x) for x in pos), f['geometry']['positions'][()])
        geometry = list(zip(atoms, positions))
    config.system.molecule = geometry
    return config

def get_N2_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.molecule_name = 'N2'
    config.bond_length = 2.068
    config.bond_length_multiple = 1.0
    config.molecule = placeholder(list)
    config.basis_set = 'aug-cc-pcvdz'
    config.basis = 'canonical'
    config.symmetry=True
    config.unit = 'Bohr'
    with config.ignore_type():
        config.set_molecule = diatomic
    return config

def get_Cr_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.molecule_name = 'Cr'
    config.molecule = [('Cr', (0., 0., 0.))]
    config.basis_set = 'cc-pvdz-dk'
    config.basis = 'canonical'
    config.symmetry=True
    config.unit = 'Angstrom'
    config.n_elec = (15, 9) # Number of α and β electrons
    config.frozen_electrons = 10 # Neon frozen core
    config.sfx2c1e = True # spin-free exact two-component one-electron integrals
    return config

def get_Cr2_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.molecule_name = 'Cr2'
    config.bond_length = 1.68
    config.bond_length_multiple = 1.0
    config.molecule = placeholder(list)
    config.basis_set = 'cc-pvdz-dk'
    config.basis = 'canonical'
    config.symmetry=True
    config.unit = 'Angstrom'
    config.frozen_electrons = 20 # Neon frozen core: 10 per atom
    config.sfx2c1e = True # spin-free exact two-component one-electron integrals
    with config.ignore_type():
        config.set_molecule = diatomic
    return config

def get_Heisenberg1d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 10
    config.J1 = 1.0
    config.pbc = True
    config.sign_rule = True
    config.total_sz = 0.0
    return config

def get_Heisenberg2d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 6
    config.Ly = 6
    config.J1 = 1.0
    config.pbc = True
    config.sign_rule = True
    config.total_sz = 0.0
    return config

def get_J1J22d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 6
    config.Ly = 6
    config.J1 = 1.0
    config.J2 = 0.5
    config.pbc = True
    config.sign_rule = True
    config.total_sz = 0
    return config

def get_Hchain_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.n_atoms = 16
    config.distance = 1.8
    config.basis_set = 'sto-6g'
    config.basis = 'canonical'
    config.symmetry = True
    config.unit = 'Bohr'
    config.molecule = placeholder(list)
    with config.ignore_type():
        config.set_molecule = chain
    return config

def get_Hring_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.n_atoms = 16
    config.distance = 1.8
    config.basis_set = 'sto-6g'
    config.basis = 'canonical'
    config.symmetry = True
    config.unit = 'Bohr'
    config.molecule = placeholder(list)
    with config.ignore_type():
        config.set_molecule = ring
    return config

def get_Hsheet_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.n_atoms = 16
    config.distance = 1.8
    config.basis_set = 'sto-6g'
    config.basis = 'canonical'
    config.symmetry = True
    config.unit = 'Bohr'
    config.molecule = placeholder(list)
    with config.ignore_type():
        config.set_molecule = sheet
    return config

def get_H2O_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.molecule = [('H', (0., 0.795, -0.454)), ('H', (0., -0.795, -0.454)), ('O', (0., 0., 0.113))]
    config.basis_set = '6-31g'
    config.basis = 'canonical'
    config.symmetry=False
    config.unit = 'Angstrom'
    return config

def get_small_molecule_config() -> ConfigDict:
    config = ConfigDict()
    config.pruning_threshold = placeholder(float)
    config.basis_set = 'sto-3g'
    config.basis = 'canonical'
    config.symmetry=False
    config.unit = 'Angstrom'
    config.molecule_name = 'H2'
    config.molecule = placeholder(list)
    with config.ignore_type():
        config.set_molecule = small_molecule
    return config

def get_Hubbard1d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 10
    config.t = 1.0
    config.U = 1.0
    config.pbc = True
    return config

def get_Hubbard2d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 4
    config.Ly = 4
    config.t = 1.0
    config.U = 1.0
    config.pbc = 'PBC-APBC'
    config.n_elec = (8, 8)
    return config

def get_config(system) -> ConfigDict:
    get_system_config = getattr(sys.modules[__name__], f"get_{system}_config")
    config = ConfigDict()
    config.system_name = system
    config.system = get_system_config()
    return config