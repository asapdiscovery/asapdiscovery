"""
Script to dock an SDF file of ligands to prepared structures.
"""
import argparse
import os
import pandas
import pickle as pkl
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covid_moonshot_ml.docking.docking import (
    build_docking_systems,
    run_docking,
)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-l", "--lig_file", required=True, help="SDF file containing ligands."
    )
    parser.add_argument(
        "-r",
        "--receptor",
        required=True,
        help="Path/glob to prepped receptor(s).",
    )
    parser.add_argument(
        "-s",
        "--sort_res",
        help="Pickle file giving compound_ids, xtal_ids, and sort_idxs.",
    )

    ## Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    ## Performance arguments
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )

    return parser.parse_args()


def main():
    args = get_args()


if __name__ == "__main__":
    main()
