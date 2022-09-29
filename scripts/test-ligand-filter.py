"""
Function to test implementation of ligand filtering
"""

import sys, os, argparse

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from covid_moonshot_ml.datasets import utils


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-f",
        "--fragalysis_dir",
        required=True,
        type=str,
        help="Path to fragalysis directory.",
    )
    parser.add_argument(
        "-c",
        "--csv_file",
        required=True,
        type=str,
        help="Path to csv file containing compound info.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    # sars_xtals = utils.parse_fragalysis_data(args.csv_file, args.fragalysis_dir)
    # sars_xtals = {
    #     compound_id: data
    #     for compound_id, data in sars_xtals.items()
    #     if data.smiles
    # }
    # print(sars_xtals)
    utils.filter_docking_inputs(smarts_queries="../data/smarts_queries.csv")


if __name__ == "__main__":
    main()
