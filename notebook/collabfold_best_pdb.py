"""
Script to find lowest rmsd value from collabfold structures.

Minimal example usage:
python collabfold_best_pdb.py \
-t "/Users/choderalab/asapdiscovery/temp_storage/8e6e.pdb" \
-r "/Users/choderalab/asapdiscovery/temp_storage/collabfold_results/"

"""

from asapdiscovery.data.openeye import load_openeye_pdb
from asapdiscovery.modeling.modeling import superpose_molecule
from openeye import oechem
from pathlib import Path
import os
import glob
import argparse

# Arguments define the input file and the output file
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-t", required=True, help="File path to pdb of template molecule."
    )
    parser.add_argument(
        "-r", required=True, help="Path to directory of results from collabfold."
    )
    parser.add_argument(
        "-c",
        default="A",
        help="Reference chain to align to."
    )
    parser.add_argument(
        "-m",
        default="A",
        help="Mobile chain to use for alignment (the whole molecule will move as well though)."
    )
    return parser.parse_args()


# Make a list of all pdb files within the directory
def find_pdb_files(directory):
    pdb_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(root, file))
    return pdb_files

def find_pdb_low_rmsd(template, results, ref_chain="A", mobile_chain="A"):
    """
    Find the pdb with the lowest rmsd value after aligning with template.

    Parameters
    ----------
    template : file path
        File path to reference molecule to align to.
    results : path
        Directory to molecule to align.
    ref_chain : Reference chain to align to
    mobile_chain : Mobile chain to use for alignment (the whole molecule will move as well though)

    Returns
    -------
    file path
        Path to best aligned molecule.
    float
        lowest RMSD between oechem.OEGraphMol template molecule and new aligned
        oechem.OEGraphMol with best match after alignment.
    """

    # Load in template file
    ref = load_openeye_pdb(template)
    lowest = 100
    the_pdb = ''
    # Iterate through the different structures generated and calculate RMSD
    pdb_files = find_pdb_files(results)
    for pdb_file in pdb_files:
        fit = load_openeye_pdb(pdb_file)
        rmsd = superpose_molecule(ref,fit)[1]
        if rmsd < lowest:
            lowest = rmsd
            the_pdb = pdb_file
    # Return the pdb file with the lowest rmsd value
    return the_pdb, lowest

def main():
    args = get_args()
    pdb, lowest = find_pdb_low_rmsd(args.t, args.r, args.c, args.m)
    print(pdb)
    print(lowest)


if __name__ == "__main__":
    main()
