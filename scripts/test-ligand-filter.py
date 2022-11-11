"""
Function to test implementation of ligand filtering
"""

import sys, os, argparse

import asap_datasets.datasets.fragalysis

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from asap_datasets.datasets import utils


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
    parser.add_argument(
        "-s",
        "--smarts_queries",
        default="../data/smarts_queries.csv",
        type=str,
        help="Path to csv file containing smarts queries.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # First, parse the fragalysis directory and
    sars_xtals = asap_datasets.datasets.fragalysis.parse_fragalysis_data(
        args.csv_file, args.fragalysis_dir
    )

    # For the compounds for which we have smiles strings, get a dictionary mapping the Compound_ID to the smiles
    cmp_to_smiles_dict = {
        compound_id: data.smiles
        for compound_id, data in sars_xtals.items()
        if data.smiles
    }

    # Filter based on the smiles using this OpenEye function
    filtered_inputs = utils.filter_docking_inputs(
        smarts_queries=args.smarts_queries,
        docking_inputs=cmp_to_smiles_dict,
    )

    # Get a new dictionary of sars xtals based on the filtered inputs
    print(filtered_inputs)
    sars_xtals_filtered = {
        compound_id: data
        for compound_id, data in sars_xtals.items()
        if compound_id in filtered_inputs
    }
    print(sars_xtals_filtered)
    print(len(sars_xtals_filtered))


if __name__ == "__main__":
    main()
