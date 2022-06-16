from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from numpy import place


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
    config.sign_rule = (True, False)
    config.total_sz = 0
    return config

def get_J1J22d_config() -> ConfigDict:
    config = ConfigDict()
    config.Lx = 6
    config.Ly = 6
    config.J1 = 1.0
    config.J2 = 0.5
    config.pbc = True
    config.sign_rule = (True, False)
    config.total_sz = 0
    return config

def get_Hchain_config() -> ConfigDict:
    config = ConfigDict()
    config.n_atoms = 10
    config.distance = 1.8
    config.atom = 'H'
    config.basis_set = 'sto-6g'
    config.basis = 'canonical'
    config.symmetry = True
    config.unit = 'Bohr'
    return config

def get_H2O_config() -> ConfigDict:
    config = ConfigDict()
    config.atom = [('H', (0., 0.795, -0.454)), ('H', (0., -0.795, -0.454)), ('O', (0., 0., 0.113))]
    config.basis_set = '6-31g'
    config.basis = 'canonical'
    config.symmetry=False
    config.unit = 'Angstrom'
    return config