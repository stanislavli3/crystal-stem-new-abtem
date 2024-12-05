import os
import pathlib

import abtem
import ase
import dotenv
import matplotlib.pyplot as plt
from abtem import GridScan
from mp_api.client import MPRester

from image import save_as_image
from specimen import make_crystal_slab


# Load the Materials Project API key
dotenv.load_dotenv()
MATERIALS_PROJECT_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")

# Set up abTEM to use the GPU
abtem.config.set({"device": "gpu", "fft": "fftw"})

# Configuration
image_width_nm = 20   # field of view of simulated images, in nanometers
image_width_px = 100  # width of simulated images, in pixels
save_specimens_to_db = True
save_simulation_results = True
save_simulation_images = True
show_plots = True

# Specimen dimensions in nanometers, angles in degrees
offset = 5   # scan offset (inward from specimen edges) for simulation
specimen_width = image_width_nm + 2*offset       # nanometers
specimen_thickness = 10     # nanometers
zone_axis_rotation = 0      # degrees

# Materials to simulate
#   --> You can add more materials by getting the mp ID numbers from the
#       Materials Explorer at materialsproject.org and adding it to this list.
materials = [
    # "mp-23",  # Ni
    "mp-1194082",  # Ni6W6C
    "mp-30811",  # Ni4W
    # "mp-91",  # W (cubic)
    # "mp-2",         # Pd (experimental)
    # "mp-81",        # Au (experimental)
    "mp-134",       # Al (experimental Fm3̅m)
    # "mp-696746",    # B4C (experimental R3̅m)
    "mp-1143",      # Al2O3 (experimental R3̅c)
    "mp-3536",      # MgAl2O4 (experimental Fd3̅m1)
]

# Zone axes to simulate
#   --> You can add more to this list, e.g., (1, 1, 2)
zone_axes = [
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
]

for mpid in materials:
    try:
        with MPRester(MATERIALS_PROJECT_API_KEY) as mpr:
            print("Getting data from MP API...")
            doc = mpr.materials.summary.get_data_by_id(mpid)
            structure = doc.structure
            formula = doc.formula_pretty

            # NOTE: If desired, you can save the Materials Project summary doc
            # as a file locally for later use (to avoid needing to call the API
            # to get crystal information), e.g. see here:
            # https://matsci.org/t/save-the-summarydoc-as-a-json-file/45409/8

    except Exception as e:
        print(e)

    for zone_axis in zone_axes:

        # strings to use in filenames when saving various results
        zone_axis_string = "".join([str(idx) for idx in zone_axis])
        specimen_string = f"{mpid}_{formula}_{zone_axis_string}"

        print(f"Creating STEM specimen for {specimen_string}...")

        specimen_dimensions = (specimen_width, specimen_width, specimen_thickness)

        specimen = make_crystal_slab(
                                    material_structure=structure,
                                    zone_axis=zone_axis,
                                    slab_dimensions=specimen_dimensions,
                                    use_conventional_unit_cell=True,
                                    grow_slab_to_fit_unit_cell=False,
                                    zone_axis_rotation_angle=zone_axis_rotation,
                                )
        if save_specimens_to_db:
            directory = "saved_specimens"
            filename = f"{specimen_string}.db"
            # create directory if necessary
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(directory, filename)
            ase.io.write(file_path, images=specimen)

            # NOTE: You can re-load the specimen into an Atoms object using
            #       `specimen = ase.io.read(filename)`.  This way, you wouldn't have
            #        to call the Materials Project API or re-calculate the specimen
            #        with `make_crystal_slab` as above.
            #        ase.read.io docs: https://wiki.fysik.dtu.dk/ase/ase/io/io.html

        # Rename specimen to "atoms" because it's an ASE Atoms object,
        # and it makes it easier to think about it this way.
        atoms = specimen

        atoms.center(axis=2, vacuum=2)

        if show_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            abtem.show_atoms(atoms, plane="xy", ax=ax1, title="Beam view")
            abtem.show_atoms(atoms, plane="xz", ax=ax2, title="Side view")
            plt.show()

        # abTEM simulation code
        # docs: https://abtem.readthedocs.io/en/latest/user_guide/examples/notebooks/stem_quickstart.html

        frozen_phonons = abtem.FrozenPhonons(atoms, 4, sigmas=0.1)

        potential = abtem.Potential(
            frozen_phonons,
            # IMPORTANT -- SAMPLING AFFECTS SPEED & THE POSSIBLE INTEGRATION ANGLES
            #              OF THE DETECTORS.  ALTERNATIVELY, CAN USE `gpts` instead of
            #              of `sampling`, but I don't recommend because if you change
            #              the specimen width, then the actuall sampling changes if
            #              `gpts` remains constant, so the allowable detector angles
            #              also change.
            sampling=0.025,
            slice_thickness=2,
        )

        s_matrix = abtem.SMatrix(
            potential=potential,
            # IMPORTANT -- INTERPOLATION AFFECTS SPEED & QUALITY
            #    in general...  interpolation=4 is slower and high quality
            #                   interpolation=6 is OK
            #                   interpolation=8 is fast and low
            #    ...but it depends on the size of the specimen, the elements in it, etc.
            interpolation=4,
            energy=200e3,
            semiangle_cutoff=30,
        )

        # spherical aberration
        # Cs = 8e-6 * 1e10
        Cs = 0
        ctf = abtem.CTF(Cs=Cs, energy=s_matrix.energy)

        detectors = abtem.FlexibleAnnularDetector()

        # It's optional to make a GridScan, but necessary if you want to determine
        # how many sampling points/pixels are in the final images
        start = [offset, offset]
        end = [specimen_width - offset, specimen_width - offset]
        sampling = image_width_nm/image_width_px
        scan = GridScan(start=start, end=end, sampling=sampling)

        flexible_measurement = s_matrix.scan(scan=scan, detectors=detectors, ctf=ctf)
        print("Computing simulation...")
        flexible_measurement.compute()

        bf_measurement = flexible_measurement.integrate_radial(0,
                                                               s_matrix.semiangle_cutoff)
        maadf_measurement = flexible_measurement.integrate_radial(45, 125)
        haadf_measurement = flexible_measurement.integrate_radial(70, 150)

        if show_plots:
            # Unfiltered measurements
            measurements = abtem.stack(
                [bf_measurement, maadf_measurement, haadf_measurement],
                ("BF", "MAADF", "HAADF")
            )
            measurements.show(
                explode=True,
                figsize=(14, 5),
                cbar=True,
            )
            plt.show()

            # Filtered measurements with Gaussian applied
            filtered_measurements = measurements.gaussian_filter(0.35)
            filtered_measurements.show(
                explode=True,
                figsize=(14, 5),
                cbar=True,
            )
            plt.show()

        if save_simulation_results:
            # Save abTEM results to a .zarr archive
            # You can read the results from .zarr with `abtem.array.from_zarr()`
            # see docs: https://abtem.readthedocs.io/en/latest/reference/api/_autosummary/abtem.array.from_zarr.html?highlight=from_zarr#from-zarr

            directory = "saved_simulations"

            # create directory if necessary
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

            def get_file_path(image_type):
                filename = f"{specimen_string}_{image_type}.zarr"
                file_path = os.path.join(directory, filename)
                return file_path

            bf_measurement.to_zarr(get_file_path("BF"))
            maadf_measurement.to_zarr(get_file_path("MAADF"))
            haadf_measurement.to_zarr(get_file_path("HAADF"))

        if save_simulation_images:
            directory = "saved_images"
            # create directory if necessary
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

            def get_file_path(image_type):
                filename = f"{specimen_string}_{image_type}.png"
                file_path = os.path.join(directory, filename)
                return file_path

            save_as_image(bf_measurement, get_file_path("BF"))
            save_as_image(maadf_measurement, get_file_path("MAADF"))
            save_as_image(haadf_measurement, get_file_path("HAADF"))

        print("Finished.")
