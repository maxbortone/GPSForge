import os
import sys
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, "../"))
import dataclasses
import datetime
import numpy as np
from absl import app
from absl import flags
from absl import logging
from typing import Iterable, Union, Any
from pyscf import scf, gto, lo, lib
from pyscf.dft import numint
from ar_qgps.configs.common import resolve
from ar_qgps.systems import build_molecule
from ml_collections import config_flags


"""
Code for rendering orbitals and densities in VMD.
Inspired by https://github.com/obackhouse/pyscf2vmd/tree/master/pyscf2vmd
"""

TACHYON_PATH = os.environ.get("TACHYON_PATH", "/usr/local/lib/vmd/tachyon_LINUXAMD64") # "/home/max/miniconda3/envs/qgps-latest/lib/tachyon_LINUXAMD64"
VMD_PATH = os.environ.get("VMD_PATH", "/usr/local/bin/vmd") # "/home/max/miniconda3/envs/qgps-latest/bin/vmd"


class CubeFile:
    """Context manager to instantiate the cube file.
    """

    def __init__(
        self,
        mol: gto.Mole,
        filename: str = "input.cube",
        gridsize=(100, 100, 100),
        resolution=None,
        title=None,
        comment=None,
        fmt="%13.5E",
        crop=None,
    ):
        self.mol = mol
        self.filename = filename

        # Make user aware of different behavior of resolution, compared to pyscf.tools.cubegen
        if resolution is not None and resolution < 1:
            logging.warning(mol, "Warning: resolution is below 1/Bohr. Recommended values are 5/Bohr or higher.")

        self.a, self.origin = self.get_box_and_origin()
        if crop is not None:
            a = self.a.copy()
            norm = np.linalg.norm(self.a, axis=1)
            a[0] -= (crop.get("a0", 0) + crop.get("a1", 0)) * self.a[0] / norm[0]
            a[1] -= (crop.get("b0", 0) + crop.get("b1", 0)) * self.a[1] / norm[1]
            a[2] -= (crop.get("c0", 0) + crop.get("c1", 0)) * self.a[2] / norm[2]
            self.origin += crop.get("a0", 0) * self.a[0] / norm[0]
            self.origin += crop.get("b0", 0) * self.a[1] / norm[1]
            self.origin += crop.get("c0", 0) * self.a[2] / norm[2]
            self.a = a
        # Use resolution if provided, else gridsize
        if resolution is not None:
            self.nx = min(np.ceil(abs(self.a[0, 0]) * resolution).astype(int), 192)
            self.ny = min(np.ceil(abs(self.a[1, 1]) * resolution).astype(int), 192)
            self.nz = min(np.ceil(abs(self.a[2, 2]) * resolution).astype(int), 192)
        else:
            self.nx, self.ny, self.nz = gridsize
        self.title = title or "<title>"
        self.comment = comment or ("Generated with GPSKet")
        self.fmt = fmt
        self.coords = self.get_coords()

        self.fields = []

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.write()

    @property
    def ncoords(self):
        """Number of grod points."""
        return self.nx * self.ny * self.nz

    @property
    def nfields(self):
        """Number of datasets (orbitals + density matrices)."""
        return len(self.fields)
    
    def get_box_and_origin(self):
        coord = self.mol.atom_coords()
        margin = 3.0
        extent = np.max(coord, axis=0) - np.min(coord, axis=0) + 2 * margin
        box = np.diag(extent)
        origin = np.asarray(np.min(coord, axis=0) - margin)
        return box, origin
    
    def get_coords(self):
        xs = np.arange(self.nx) / (self.nx - 1)
        ys = np.arange(self.ny) / (self.ny - 1)
        zs = np.arange(self.nz) / (self.nz - 1)
        coords = lib.cartesian_prod([xs, ys, zs])
        coords = np.dot(coords, self.a)
        coords = np.asarray(coords, order="C") + self.origin
        return coords

    def add_orbital(self, coeff, dset_idx=None):
        """Add one or more orbitals to the cube file.

        Arguments
        ---------
        coeff : (N) or (N,M) array
            AO coefficients of orbitals. Supports adding a single orbitals,
            where `coeff` is a one-dimensional array, or multiple orbitals,
            in which case the second dimension of `coeff` labels the orbitals.
        dset_idx : int, optional
            Dataset index of orbital(s). In the application, the orbitals will
            be labelled as 'Orbital <dset_idx>' or similar. If set to `None`,
            the smallest unused, positive integer will be used. Default: None.
        """
        coeff = np.array(coeff)  # Force copy
        if coeff.ndim == 1:
            coeff = coeff[:, np.newaxis]
        assert coeff.ndim == 2
        for i, c in enumerate(coeff.T):
            idx = dset_idx + i if dset_idx is not None else None
            self.fields.append((c, "orbital", idx))

    def add_density(self, dm):
        """Add one density to the cube file.

        Arguments
        ---------
        dm : (N,N) array
            Density-matrix in AO-representation.
        """
        dm = np.array(dm)  # Force copy
        assert dm.ndim == 2
        self.fields.append((dm, "density", 0))

    def write(self, filename=None):
        filename = filename or self.filename
        # Create directories if necessary
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Get dataset IDs
        dset_ids = []
        for field, ftype, fid in self.fields:
            if fid is None:
                if dset_ids:
                    fid = np.max(dset_ids) + 1
                else:
                    fid = 1
            dset_ids.append(fid)

        self.write_header(filename, dset_ids=dset_ids)
        self.write_fields(filename)

    def write_header(self, filename, dset_ids=None):
        """Write header of cube-file."""
        if self.nfields > 1 and dset_ids is None:
            dset_ids = range(1, self.nfields + 1)
        with open(filename, "w") as f:
            f.write("%s\n" % self.title)
            f.write("%s\n" % self.comment)
            if self.nfields > 1:
                f.write("%5d" % -self.mol.natm)
            else:
                f.write("%5d" % self.mol.natm)
            f.write("%12.6f%12.6f%12.6f" % tuple(self.origin))
            if self.nfields > 1:
                f.write("%5d" % self.nfields)
            f.write("\n")
            # Lattice vectors
            f.write("%5d%12.6f%12.6f%12.6f\n" % (self.nx, *(self.a[0] / (self.nx - 1))))
            f.write("%5d%12.6f%12.6f%12.6f\n" % (self.ny, *(self.a[1] / (self.ny - 1))))
            f.write("%5d%12.6f%12.6f%12.6f\n" % (self.nz, *(self.a[2] / (self.nz - 1))))
            # Atoms
            for atm in range(self.mol.natm):
                sym = self.mol.atom_symbol(atm)
                f.write("%5d%12.6f" % (gto.charge(sym), 0.0))
                f.write("%12.6f%12.6f%12.6f\n" % tuple(self.mol.atom_coords()[atm]))
            # Data set indices
            if self.nfields > 1:
                f.write("%5d" % self.nfields)
                for i in range(self.nfields):
                    f.write("%5d" % dset_ids[i])
                f.write("\n")

    def write_fields(self, filename):
        """Write voxel data of registered fields in `self.fields` to cube-file."""
        blksize = min(self.ncoords, 8000)
        with open(filename, "a") as f:
            # Loop over x,y,z coordinates first, then fields!
            for blk0, blk1 in lib.prange(0, self.ncoords, blksize):
                data = np.zeros((blk1 - blk0, self.nfields))
                blk = np.s_[blk0:blk1]
                intor =  "GTOval"
                ao = self.mol.eval_gto(intor, self.coords[blk])
                for i, (field, ftype, _) in enumerate(self.fields):
                    if ftype == "orbital":
                        data[:, i] = np.dot(ao, field)
                    elif ftype == "density":
                        data[:, i] = numint.eval_rho(self.mol, ao, field)
                    else:
                        raise ValueError("Unknown field type: %s" % ftype)
                data = data.flatten()
                for d0, d1 in lib.prange(0, len(data), 6):
                    f.write(((d1 - d0) * self.fmt + "\n") % tuple(data[d0:d1]))

@dataclasses.dataclass
class Options:
    """Options for the VMD orbital plot.
    """

    # Display settings
    projection: str = "Stereographic"
    depthcue: bool = False
    axes: bool = False
    carbon_color: Iterable[float] = (0.5, 0.5, 0.5)
    background_color: Iterable[float] = (1.0, 1.0, 1.0)
    rotate: Union[Iterable[float], str] = (0.0, 0.0, 0.0)  # If str, load from visualisation state
    lights: Iterable[bool] = (True, True, False, False)
    shadows: bool = True
    zoom: float = 1.0

    # Molecule settings
    bond_radius: float = 0.15
    atom_radius: float = None  # set to None for 'licorice' layout
    mol_res: int = 200
    mol_material: str = "AOChalky"

    # Density settings
    show_density: bool = True
    den_isovalue: float = 0.05
    den_material: str = "AOShiny"
    den_color_pos: Iterable[float] = (0.84, 0.15, 0.16)  # Red color

    # Orbital settings
    show_orbitals: bool = True
    orb_isovalue: float = 0.05
    orb_material: str = "AOShiny"
    orb_color_pos: Iterable[float] = (0.122, 0.467, 0.706)  # Matplotlib blue
    orb_color_neg: Iterable[float] = (1.000, 0.498, 0.055)  # Matplotlib orange

    # Render settings
    render_res: Iterable[float] = (4000, 3000)
    ambient_occlusion: bool = True
    ao_ambient: float = 0.7
    ao_direct: float = 0.2

    # Material settings
    material_settings: dict = dataclasses.field(
        default_factory=lambda: {
            "AOShiny": {
                "specular": 1.0,
                "ambient": 0.5,
                "shininess": 0.7,
            }
        }
    )

    # File settings
    input_file: str = "input.cube"
    output_name: str = "output"
    keep_vmd_input: bool = False
    quit_vmd: bool = True
    tachyon_path: str = TACHYON_PATH
    vmd_path: str = VMD_PATH
    convert_to_png: bool = True

    # Application settings
    interactive: bool = False

    # Other settings
    other_commands: Iterable[Any] = tuple()


class Plotter:
    """Class to handle the VMD plotting.
    """

    def __init__(self, options=None, **kwargs):
        if options is None:
            options = Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)
        self.ndsets = 1

        if not os.path.exists(VMD_PATH):
            raise FileNotFoundError(f"VMD executable not found at {VMD_PATH}. Please specify a correct path")

    def parse_cubefile(self):
        """Parse the cube file to read the number of datasets.
        """

        with open(self.options.input_file, "r") as f:
            i = 0
            while i <= 2:
                line = f.readline()
                i += 1
            columns = line.strip().split(' ')
            if int(columns[0]) < 0:
                natoms = abs(int(columns[0]))
                while i <= natoms+6:
                    line = f.readline()
                    i += 1
                columns = line.strip().split(' ')
                ndsets = int(columns[0])
                self.ndsets = ndsets
    
    def init_input(self):
        """Initialise the VMD input file.
        """

        with open("input.vmd", "w") as f:
            f.write("# VMD input file generated using pyscf2vmd\n")
            f.write("# Input file: %s\n" % self.options.input_file)
            f.write("# Generated at %s\n\n" % datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    def clean(self):
        """Clean up input file.
        """

        if not self.options.keep_vmd_input:
            os.system("rm -f %s input.vmd" % self.options.input_file)

        os.system("rm -f vmdscene.dat")
        os.system("mv vmdscene.dat.tga %s.tga" % self.options.output_name)

        if self.options.convert_to_png:
            os.system("convert %s.tga %s.png" % (self.options.output_name, self.options.output_name))
            os.system("rm -f %s.tga" % self.options.output_name)

    def add_command(self, cmd):
        """Add command to the input file.
        """

        with open("input.vmd", "a") as f:
            f.write(cmd + "\n")

    def launch_vmd(self):
        """Launch VMD with the input file.
        """

        if self.options.interactive:
            os.system("%s -e input.vmd" % self.options.vmd_path)
        else:
            os.system("%s -dispdev text -e input.vmd" % self.options.vmd_path)

    def write_input(self):
        """Write the input file.
        """

        assert os.path.isfile(self.options.input_file)

        self.add_command("# Display settings")
        self.add_command("display projection %s" % self.options.projection)
        self.add_command("display depthcue %s" % ["off", "on"][self.options.depthcue])
        if not self.options.axes:
            self.add_command("axes location off")
        if self.options.ambient_occlusion:
            self.add_command("display ambientocclusion on")
            self.add_command("display aoambient %f" % self.options.ao_ambient)
            self.add_command("display aodirect %f" % self.options.ao_direct)
        for i, light in enumerate(self.options.lights):
            self.add_command("light %d %s" % (i, ["off", "on"][light]))
        self.add_command("display shadows %s" % ["off", "on"][self.options.shadows])
        self.add_command("\n")

        self.add_command("# Load molecule")
        self.add_command("mol delete all")
        self.add_command("mol new %s type %s" % (self.options.input_file, self.options.input_file.split(".")[-1]))
        self.add_command("\n")

        self.add_command("# Material settings")
        for mat, val in self.options.material_settings.items():
            for prop, value in val.items():
                self.add_command("material change %s %s %f" % (prop, mat, value))
        self.add_command("\n")

        self.add_command("# Add the molecule representation")
        self.add_command("mol delrep 0 top")
        if self.options.atom_radius is None:
            self.add_command("mol representation licorice %f %d %d" % (self.options.bond_radius, self.options.mol_res, self.options.mol_res))
        else:
            self.add_command("mol representation CPK %f %f %d %d" % (self.options.atom_radius * 5, self.options.bond_radius * 5, self.options.mol_res, self.options.mol_res))
        self.add_command("mol color Name")
        self.add_command("mol material %s" % self.options.mol_material)
        self.add_command("mol addrep top")
        self.add_command("\n")

        if self.options.show_density is not None:
            self.add_command("# Add the density representation for dataset 0")
            self.add_command("mol representation Isosurface %f 0 0 0 1 1" % self.options.den_isovalue)
            self.add_command("mol selection all")
            self.add_command("mol material %s" % self.options.den_material)
            self.add_command("color change rgb 28 %f %f %f" % tuple(self.options.den_color_pos))
            self.add_command("mol color ColorID 28")
            self.add_command("mol addrep 0")
            self.add_command("\n")

        if self.options.show_orbitals is not None:
            start = 1 if self.options.show_density else 0
            for dset_idx in range(start, self.ndsets):
                self.add_command("# Add the positive orbital representation for dataset %d" % dset_idx)
                self.add_command("mol representation Isosurface %f %d 0 0 1 1" % (self.options.orb_isovalue, dset_idx))
                self.add_command("mol selection all")
                self.add_command("mol material %s" % self.options.orb_material)
                self.add_command("color change rgb 30 %f %f %f" % tuple(self.options.orb_color_pos))
                self.add_command("mol color ColorID 30")
                self.add_command("mol addrep 0")
                self.add_command("\n")

                self.add_command("# Add the negative orbital representation for dataset %d" % dset_idx)
                self.add_command("mol representation Isosurface %f %d 0 0 1 1" % (-self.options.orb_isovalue, dset_idx))
                self.add_command("mol selection all")
                self.add_command("mol material %s" % self.options.orb_material)
                self.add_command("color change rgb 29 %f %f %f" % tuple(self.options.orb_color_neg))
                self.add_command("mol color ColorID 29")
                self.add_command("mol addrep 0")
                self.add_command("\n")

        if self.options.carbon_color:
            self.add_command("# Change colours")
            self.add_command("color change rgb 31 %f %f %f" % tuple(self.options.carbon_color))
            self.add_command("color Name C orange2")
            self.add_command("color Type C orange2")
        if self.options.background_color:
            self.add_command("color change rgb 32 %f %f %f" % tuple(self.options.background_color))
            self.add_command("color Display Background orange3")
        if self.options.carbon_color or self.options.background_color:
            self.add_command("\n")

        self.add_command("# Rotate the view")
        if isinstance(self.options.rotate, (list, tuple)):
            self.add_command("rotate x by %f" % self.options.rotate[0])
            self.add_command("rotate y by %f" % self.options.rotate[1])
            self.add_command("rotate z by %f" % self.options.rotate[2])
        elif self.options.rotate is not None:
            with open(self.options.rotate, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if lines[i].startswith("set viewpoints"):
                        start = i
                        break
                for i in range(start, len(lines)):
                    if lines[i].startswith("unset topmol"):
                        end = i+1
                        break
                self.add_command("set viewplist {}")
                self.add_command("set fixedlist {}")
                for line in lines[start:end]:
                    self.add_command(line.strip())
        self.add_command("scale by %f" % self.options.zoom)
        self.add_command("\n")

        if len(self.options.other_commands):
            self.add_command(self.options.other_commands)
            self.add_command("\n")

        self.add_command("# Render the image")
        self.add_command("render Tachyon vmdscene.dat "
            + "\"%s\" " % self.options.tachyon_path
            + "-aasamples 12 %s -format TARGA "
            + "-res %d %d " % self.options.render_res
            + "-o %s.tga"
        )
        self.add_command("\n")

        if self.options.quit_vmd:
            self.add_command("# cya")
            self.add_command("quit")

    def run(self):
        """Run the workflow.
        """

        self.parse_cubefile()
        self.init_input()
        self.write_input()
        self.launch_vmd()
        self.clean()

FLAGS = flags.FLAGS

main_orbital = flags.DEFINE_integer('main_orbital', 0, 'Index of the main orbital for which to render the density of closest coupled orbitals')
K = flags.DEFINE_integer('K', 5, 'Number of closest coupled orbitals')
additional_orbitals = flags.DEFINE_list('additional_orbitals', [], 'List of additional orbital indices to render')
output_file = flags.DEFINE_string('output_file', 'output', 'Output file name')
rotate = flags.DEFINE_list('rotate', [-45.0, 0.0, 0.0], 'Rotation angles')
density_isovalue = flags.DEFINE_float('density_isovalue', 0.03, 'Isovalue for the density')
orbital_isovalue = flags.DEFINE_float('orbital_isovalue', 0.03, 'Isovalue for the orbitals')
zoom = flags.DEFINE_float('zoom', 1.0, 'Zoom value for VMD')


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

    # Choose closest coupled orbitals
    orbital_indices = top_k_indices[int(main_orbital.value)]
    main_orbitals = basis[:, orbital_indices]

    # Calculate density of closest coupled orbitals
    density = np.einsum('ia,ja->ij', main_orbitals, main_orbitals.conj())

    # Select additional orbitals
    if len(additional_orbitals.value) > 0:
        orbital_indices = list(map(int, additional_orbitals.value))
        orbitals = basis[:, orbital_indices]

    # Generate cube file
    with CubeFile(mol, filename="input.cube") as f:
        f.add_density(density)
        if len(additional_orbitals.value) > 0:
            f.add_orbital(orbitals)

    # Set options
    material_settings = {
        "AOShiny": {
            "specular": 1.0,
            "ambient": 0.5,
            "shininess": 0.7,
            "opacity": 0.7
        }
    }
    options = dict(
        render_res=(1920, 1080),
        convert_to_png=True,
        den_isovalue=density_isovalue.value,
        orb_isovalue=orbital_isovalue.value,
        rotate=rotate.value,
        zoom=zoom.value,
        keep_vmd_input=False,
        tachyon_path=TACHYON_PATH,
        vmd_path=VMD_PATH,
        output_name=f"{output_file.value}_{main_orbital.value}",
        material_settings=material_settings
    )

    # Plot the orbital
    plotter = Plotter(**options)
    plotter.run()


if __name__ == '__main__':
    app.run(main)