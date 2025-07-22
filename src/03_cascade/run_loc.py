# --- Import Libraries ---
import os
import numpy as np

from mpi4py import MPI
from lammps import lammps, PyLammps

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(__file__)

POTENTIAL_PATH = os.path.join('potentials', 'malerba.fs')
INPUT_PATH = os.path.join(ROOT_DIR, '01_input_generation', 'output', 'quadrupole.lmp')
THERMO_DUMP_DIR = os.path.join(ROOT_DIR, 'src', PARENT_DIR, 'output_thermo')
CASCADE_DUMP_DIR = os.path.join(ROOT_DIR, 'src', PARENT_DIR, 'output_cascade')

# --- Parameters ---
TEMPERATURE = 100
DT = 0.001 # Timestep for thermalisation

def main():

    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: 
        os.makedirs(THERMO_DUMP_DIR, exist_ok=True)
        os.makedirs(CASCADE_DUMP_DIR, exist_ok=True)

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

    # Thermalise the system for 10ps
    L.velocity('fe_atoms', 'create', 'TEMPERATUER', np.random.randint(1, 1e6), 'rot', 'yes', 'dist', 'gaussian') # Give atoms a bunch of energy

    L.fix(1, 'fe_atoms', 'npt', TEMPERATURE, TEMPERATURE, 100.0*DT, 'iso', 0.0, 0.0, 1000.0*DT)



    # Run cascade simulation

    return None

# --- Entrypoint ---

if __name__ == "__main__":

    main()