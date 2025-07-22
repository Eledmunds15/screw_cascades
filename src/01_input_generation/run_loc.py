# --- Import Libraries ---
import os
import numpy as np
from matscipy.dislocation import get_elastic_constants, BCCScrew111Dislocation, Quadrupole
from matscipy.calculators.eam import EAM

from ase.io import write

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(__file__)

POTENTIAL_PATH = os.path.join('potentials', 'malerba.fs')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

# --- Parameters ---

GLIDE_SEPARATION = 50 # Glide separation in angstroms
BOX_HEIGHT = 50 # Height of the box in planes of atoms

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    eam_calc = EAM(POTENTIAL_PATH)
    alat, C11, C12, C44 = get_elastic_constants(calculator=eam_calc, symbol='Fe', verbose='False') # Get information from the potential file for file creation
    print(f"{alat:.3f} (Angstrom), {C11:.2f}, {C12:.2f}, {C44:.2f} (GPa)") # Print information

    Fe_edge = BCCScrew111Dislocation(alat, C11, C12, C44, symbol="Fe") # Create dislocation object

    quad = Quadrupole(BCCScrew111Dislocation, alat, C11, C12, C44, symbol='Fe')

    quad_bulk_plane, quad_disloc_plane = quad.build_quadrupole(glide_separation=GLIDE_SEPARATION)

    quad_dislo_box = quad_disloc_plane.repeat((1,1,BOX_HEIGHT)) # Replicate the cylinder along the dislocation axis (z)

    print(f"Number of atoms: {len(quad_dislo_box)}") # Find the number of atoms in the sim

    write(os.path.join(OUTPUT_DIR, 'quadrupole.lmp'), quad_dislo_box, format="lammps-data", specorder=['Fe']) # Write the file out to lammps input file

    return None

# --- Entrypoint ---

if __name__ == "__main__":

    main()