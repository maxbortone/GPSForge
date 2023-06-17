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

def sheet(config):
    Lx, Ly = closest_divisors(config.system.n_atoms)
    molecule = []
    dist = config.system.distance
    for x in range(Lx):
        for y in range(Ly):
            molecule.append(('H', (x*dist, y*dist, 0.)))
    config.system.molecule = molecule
    return config

def get_Heisenberg1d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 10
    config.J1 = 1.0
    config.pbc = True
    config.sign_rule = True
    config.total_sz = 0
    return config

def get_Heisenberg2d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 6
    config.Ly = 6
    config.J1 = 1.0
    config.pbc = True
    config.sign_rule = True
    config.total_sz = 0
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

def get_Hsheet_config() -> ConfigDict:
    config = ConfigDict()
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
    config.molecule = [('H', (0., 0.795, -0.454)), ('H', (0., -0.795, -0.454)), ('O', (0., 0., 0.113))]
    config.basis_set = '6-31g'
    config.basis = 'canonical'
    config.symmetry=False
    config.unit = 'Angstrom'
    return config

def get_Hubbard1d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 10
    config.t = 1.0
    config.U = 1.0
    config.pbc = True
    return config