"""
The purpose of this script is to load in a single prepared protein-ligand system, 
solvate it, and briefly equilibrate it.

TODO: ADD EXAMPLE USAGE
"""
## Imports
import openmm
import argparse

## Parameters
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i", "--input_pdb_path", help="Path to PDB file to simulate."
    )
    parser.add_argument(
        "-o", "--output_dir", help="Output simulation directory."
    )
    return parser.parse_args()


## Functions
def load_openmm_pdb(input_pdb_path):
    pass


def save_openmm_simulation(output_dir):
    ## Save system.xml

    ## Save state.xml

    ## Save integrator.xml

    ## Save equilibrated.pdb

    pass


## Main Script
def main():
    ## Load PDB

    ## Create System

    ## Set Simulation params

    ## Create Simulation object

    ## Run Simulation
    pass
