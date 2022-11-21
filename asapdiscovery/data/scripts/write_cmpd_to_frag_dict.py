"""
The goals is to create a saved dictionary mapping the compound_ID to the fragalysis structure we are using.
i.e. `AAR-POS-0daf6b7e-1: Mpro-x1311`
The input is the compound_tracker.csv file, the output is a yaml file (default is in data/cmpd_to_frag.yaml).
This generates a required input for the `fauxalysis_from_docking.py` script.
Example Usage:
    python write_cmpd_to_frag_dict.py
        -f ~/Scientific_Projects/mers-drug-discovery/Mpro-paper-ligand/extra_files/Mpro_compound_tracker_csv.csv

"""
import argparse, os, sys, yaml

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from asapdiscovery.data.utils import (
    get_compound_id_xtal_dicts,
    parse_fragalysis_data,
)


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-f",
        "--frag_csv",
        required=True,
        help="Fragalysis crystal structure compound_tracker.csv file.",
    )
    parser.add_argument(
        "-o",
        "--out_yaml",
        default=os.path.join(
            repo_path,
            "data",
            "cmpd_to_frag.yaml",
        ),
        help="Path to output yaml file.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    frag_dir = os.path.dirname(args.frag_csv)

    ## First, parse the fragalysis directory into a dictionary of
    ##  CrystalCompoundData
    ## TODO: Update the parse_fragalysis_data function to use the metadata.csv file as that contains all the structures
    sars_xtals = parse_fragalysis_data(args.frag_csv, frag_dir)

    ## Get dict mapping crystal structure id to compound id
    compound_id_dict = {
        cmpd: xtals[0]
        for cmpd, xtals in get_compound_id_xtal_dicts(sars_xtals.values())[
            0
        ].items()
        if cmpd
    }

    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(compound_id_dict, f)


if __name__ == "__main__":
    main()
