import argparse
import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
from covid_moonshot_ml.datasets import pdb

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", help="Directory name to put the structures")
    parser.add_argument(
        "-y", default="mers-structures.yaml", help="MERS structures yaml file"
    )
    parser.add_argument(
        "-r", default=None, help="Path to pdb reference file to align to"
    )
    parser.add_argument("-n", default=None, help="Name of reference")
    return parser.parse_args()


def main():
    args = get_args()
    pdb_list = pdb.load_pdbs_from_yaml(args.y)
    pdb.download_PDBs(pdb_list, args.d)
    sel_dict = {"chainA_protein": "chain A and polymer.protein"}
    pdb.align_all_pdbs(pdb_list, args.d, args.r, args.n, sel_dict)


if __name__ == "__main__":
    main()
