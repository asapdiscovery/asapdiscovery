"""
The purpose of this script is to make a multi-ligand SDF file which would be an input to the run_docking_oe.py script.
Currently, the point is to process the fragalysis dataset.
Example Usage:
    python filter_mpro_sdf.py \
    -csv /Users/alexpayne/Scientific_Projects/mers-drug-discovery/mpro-paper-ligand/extra_files/Mpro_compound_tracker_csv.csv \
    -s /Users/alexpayne/Scientific_Projects/mers-drug-discovery/mpro-paper-ligand/aligned/ \
    -o /Volumes/Rohirrim/local_test/sars_docking/fragalysis_correct_bond_orders.sdf
"""
import argparse
import os
import sys

from openeye import oechem

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from asapdiscovery.data.fragalysis import parse_xtal
from asapdiscovery.data.openeye import save_openeye_sdfs


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-csv",
        "--xtal_csv",
        required=True,
        help="Path to fragalysis Mpro_compound_tracker_csv.csv.",
    )
    parser.add_argument(
        "-s",
        "--structure_dir",
        required=True,
        help="Path to fragalysis structure directory.",
    )
    parser.add_argument(
        "-o",
        "--sdf_fn",
        required=True,
        help="Path to output multi-object sdf file that will be created",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Parsing '{args.xtal_csv}'")
    xtal_compounds = parse_xtal(args.xtal_csv, args.structure_dir)
    print(f"Example: \n{xtal_compounds[0]}")

    ## TODO: Might want to add ability to include input positions
    ## TODO: Might also want to add more data to the output SDF files

    ## Make OEGraphMol for each compound and include some of the data
    print(f"Creating {len(xtal_compounds)} OEGraphMol objects")
    mols = []
    for c in xtal_compounds:
        new_mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(new_mol, c.smiles)
        oechem.OESetSDData(new_mol, f"SMILES", c.smiles)
        oechem.OESetSDData(new_mol, f"Dataset", c.dataset)
        oechem.OESetSDData(new_mol, f"Compound_ID", c.compound_id)
        new_mol.SetTitle(c.compound_id)
        mols.append(new_mol)

    print(f"Saving to {args.sdf_fn}")
    save_openeye_sdfs(mols, args.sdf_fn)


if __name__ == "__main__":
    main()
