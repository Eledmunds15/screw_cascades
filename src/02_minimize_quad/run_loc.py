# --- Import Libraries ---
import os

from mpi4py import MPI
from lammps import lammps, PyLammps

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(__file__)

POTENTIAL_PATH = os.path.join('potentials', 'malerba.fs')
INPUT_PATH = os.path.join(ROOT_DIR, '01_input_generation', 'output', 'quadrupole.lmp')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'src', PARENT_DIR, 'output')
DUMP_DIR = os.path.join(ROOT_DIR, 'src', PARENT_DIR, 'dump')

# --- Parameters ---

ENERGY_TOL = 1e-6
FORCE_TOL = 1e-8

def main():

    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: 
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(DUMP_DIR, exist_ok=True)

    # LAMMPS Script
    lmp = lammps()
    L = PyLammps(ptr=lmp)

    L.log(os.path.join(OUTPUT_DIR, 'log.lammps'))

    L.units('metal') # Set units style
    L.atom_style('atomic') # Set atom style

    L.command('boundary p p p') # Set the boundaries of the simulation

    L.read_data(INPUT_PATH) # Read input file

    L.pair_style('eam/fs') # Set the potential style
    L.pair_coeff('*', '*', POTENTIAL_PATH, 'Fe') # Select the potential

    L.group('fe_atoms', 'type', 1) # Group all atoms

    L.compute('peratom', 'all', 'pe/atom') # Set a compute to track the peratom energy

    L.minimize(ENERGY_TOL, FORCE_TOL, 1000, 10000) # Execute minimization

    L.write_dump('all', 'custom', os.path.join(DUMP_DIR, 'dump'), 'id', 'x', 'y', 'z', 'c_peratom') # Write a dumpfile containing atom positions and pot energies
    L.write_data(os.path.join(OUTPUT_DIR, 'quad.lmp'))

    L.close()

    return None

# --- Entrypoint ---

if __name__ == "__main__":

    main()