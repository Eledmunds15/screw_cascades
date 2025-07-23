# --- Import Libraries ---
import os
import re
import numpy as np

from mpi4py import MPI
from lammps import lammps, PyLammps

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(__file__)

POTENTIAL_PATH = os.path.join('potentials', 'malerba.fs')
INPUT_PATH = os.path.join(ROOT_DIR, '01_input_generation', 'output', 'quadrupole.lmp')
THERMO_DUMP_DIR = os.path.join(ROOT_DIR, 'src', PARENT_DIR, 'output', 'output_thermo')
CASCADE_DUMP_DIR = os.path.join(ROOT_DIR, 'src', PARENT_DIR, 'output', 'output_cascade')

# --- Parameters ---
TEMPERATURE = 100
DT = 0.001 # Timestep for thermalisation

PKA_X = 37 # X position of primary knock on atom (PKA)
PKA_Y = 62 # Y position of PKA
PKA_Z = 37 # Z position of PKA
PKA_E = 10 # Energy of PKA

def main():

    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: 
        os.makedirs(os.path.dirname(THERMO_DUMP_DIR), exist_ok=True)
        os.makedirs(THERMO_DUMP_DIR, exist_ok=True)
        os.makedirs(CASCADE_DUMP_DIR, exist_ok=True)

    # LAMMPS Script
    lmp = lammps()
    L = PyLammps(ptr=lmp)

    L.log(os.path.join(os.path.dirname(THERMO_DUMP_DIR), 'log.lammps'))

    L.units('metal') # Set units style
    L.atom_style('atomic') # Set atom style

    L.command('boundary p p p') # Set the boundaries of the simulation

    L.read_data(INPUT_PATH) # Read input file

    L.pair_style('eam/fs') # Set the potential style
    L.pair_coeff('*', '*', POTENTIAL_PATH, 'Fe') # Select the potential

    L.group('fe_atoms', 'type', 1) # Group all atoms

    L.compute('peratom', 'all', 'pe/atom') # Set a compute to track the peratom energy

    # Thermalise the system for 10ps

    L.velocity('fe_atoms', 'create', TEMPERATURE, np.random.randint(1, 1e6), 'rot', 'yes', 'dist', 'gaussian') # Give atoms a bunch of energy

    L.fix(1, 'fe_atoms', 'npt', 'temp', TEMPERATURE, TEMPERATURE, 100.0*DT, 'iso', 0.0, 0.0, 1000.0*DT)

    L.dump(1, 'all', 'custom', 20, os.path.join(THERMO_DUMP_DIR, 'dumpfile_*'), 'id', 'x', 'y', 'z', 'c_peratom')
    L.thermo(20)

    L.run(500)

    _, ref_file = find_last_dump()
    if rank == 0: print(ref_file)

    pka_id = find_atom_ID(ref_file)
    if rank == 0: print(pka_id)

    return None

# --- Functions ---

def find_last_dump(dir=THERMO_DUMP_DIR):

    if not os.path.isdir(dir):
        print(f"Error: The specified path '{dir}' is not a valid directory")
        return None
    
    files = []

    pat = re.compile(r'dumpfile_(\d+)')

    try:
        for file in os.listdir(dir):
            match = pat.match(file)
            if match:
                try:
                    timestep = int(match.group(1))
                    files.append((timestep, os.path.join(dir, file)))
                except ValueError:
                    print(f'File {file} could not be processed')
                    continue
    except OSError as e:
        print(f"Error accessing directory '{dir}: {e}'")
        return None
    
    if not files:
        print(f"Directory {dir} empty...")
        return None

    files.sort(key=lambda x: x[0])

    # print(f"Last File: {files[-1][1]} | Timestep: {files[-1][0]}")
    return files[-1]

def find_atom_ID(ref_file, pka_x=PKA_X, pka_y=PKA_Y, pka_z=PKA_Z):

    if not os.path.exists(ref_file):
        print(f"Error: Reference file '{ref_file}' not found.")
        return None

    try:
        with open(ref_file, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading file '{ref_file}': {e}")
        return None

    header_line = -1
    header_labels = []

    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            header_line = i
            # Split the header to get individual labels (e.g., 'id', 'x', 'y', 'z')
            header_labels = line.split()[2:] 
            break # Found the header, no need to continue searching

    if header_line == -1:
        print(f"Error: 'ITEM: ATOMS' section not found in '{ref_file}'.")
        return None

    # Determine the column indices for 'id', 'x', 'y', 'z'
    try:
        id_col = header_labels.index('id')
        x_col = header_labels.index('x')
        y_col = header_labels.index('y')
        z_col = header_labels.index('z')
    except ValueError as e:
        print(f"Error: Required column (id, x, y, or z) not found in 'ITEM: ATOMS' header: {e}")
        print(f"Header labels found: {header_labels}")
        return None

    min_distance_sq = float('inf') # Initialize with a very large distance
    closest_atom_id = None

    # Process atom data lines
    # Atom data starts from the line *after* the header_line
    for line_num in range(header_line + 1, len(lines)):
        line = lines[line_num].strip()
        if not line: # Skip empty lines
            continue
        
        # LAMMPS dump files can have empty lines or other ITEM headers after atom data
        # Stop if we encounter another ITEM header
        if line.startswith("ITEM:"):
            break

        parts = line.split()
        
        # Ensure enough parts to extract required data
        if len(parts) <= max(id_col, x_col, y_col, z_col):
            print(f"Warning: Skipping malformed line {line_num + 1}: '{line}'")
            continue

        try:
            atom_id = int(parts[id_col])
            atom_x = float(parts[x_col])
            atom_y = float(parts[y_col])
            atom_z = float(parts[z_col])
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse atom data from line {line_num + 1}: '{line}' - {e}")
            continue

        # Calculate squared Euclidean distance to avoid sqrt for every comparison
        # (sqrt is computationally expensive and not needed for comparison)
        distance_sq = (atom_x - pka_x)**2 + (atom_y - pka_y)**2 + (atom_z - pka_z)**2

        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            closest_atom_id = atom_id

    return closest_atom_id

# --- Entrypoint ---

if __name__ == "__main__":

    main()