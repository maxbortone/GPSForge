import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
sys.path.append(os.path.join(base_path, "../../pyscf2vmd"))
import numpy as np
from absl import app
from absl import flags
from absl import logging
from pyscf import scf, gto, lo
from ar_qgps.configs.systems import get_config
from ar_qgps.configs.common import resolve
from ar_qgps.systems import build_molecule
from ml_collections import config_flags
from pyscf2vmd import Plotter, CubeFile


FLAGS = flags.FLAGS

orbitals = flags.DEFINE_list('orbitals', [14], 'Indices of the orbitals to plot')
K = flags.DEFINE_integer('K', 5, 'Number of closest coupled orbitals')
output_file = flags.DEFINE_string('output_file', 'output', 'Output file name')
rotate = flags.DEFINE_list('rotate', [-45.0, 0.0, 0.0], 'Rotation angles for VMD')
isovalue = flags.DEFINE_float('isovalue', 0.36, 'Isovalue for VMD')


_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the system configuration.',
    lock_config=True
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    # Get config
    config = _CONFIG.value
    config = resolve(config)
    logging.info('Config: %s', config)

    # Setup molecular system
    mol = build_molecule(config.system)
    logging.info(f"Nuclear energy {mol.energy_nuc()}")

    # Run Hartree Fock
    mf = scf.RHF(mol).run()
    print('reference HF total energy =', mf.e_tot)

    # Get converged density matrix from the Hartree Fock
    dm = mf.make_rdm1()
    _, vk = mf.get_jk(mol, dm) # vk is the exchange in the AO basis

    if config.system.basis == "local-boys":
        # Transform to a local orbital basis
        loc_coeff = lo.orth_ao(mol, 'lowdin')
        localizer = lo.Boys(mol, mo_coeff=loc_coeff)
        localizer.init_guess = None
        basis = localizer.kernel()

        # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
        norb = mf.mo_coeff.shape[1]
        ovlp = mf.get_ovlp()
        assert(np.allclose(np.linalg.multi_dot((basis.T, ovlp, basis)), np.eye(norb)))

        # Rotate exchange matrix into local basis
        vk = np.linalg.multi_dot((basis.T, vk, basis))

    # Generate environment matrix of top-K closest coupled orbitals for each orbital
    top_k_indices = np.flip(np.argsort(np.abs(vk), axis=1)[:, -K.value:], axis=1)

    orbital_indices = np.array(orbitals.value, np.int32)
    for orbital_index in orbital_indices:
        # Choose orbital
        orbital = np.sum(basis[:, top_k_indices[orbital_index]], axis=1)

        # Set options
        options = dict(
            render_res=(1920, 1080),
            convert_to_png=True,
            isovalue=isovalue.value,
            rotate=rotate.value,
            keep_vmd_input=True,
            tachyon_path="/home/max/miniconda3/envs/qgps-latest/lib/tachyon_LINUXAMD64",
            vmd_path="/home/max/miniconda3/envs/qgps-latest/bin/vmd",
            output_name=f"{output_file.value}_{orbital_index}"
        )

        # Plot the orbital
        with CubeFile(mol, orbital=orbital):
            plotter = Plotter(**options)
            plotter.run()


if __name__ == '__main__':
    app.run(main)