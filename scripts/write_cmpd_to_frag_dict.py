"""
Build library of ligands from a dataset of holo crystal structures docked to a
different dataset of apo structures.
"""
import argparse, os, sys, yaml

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from covid_moonshot_ml.datasets.utils import (
    get_compound_id_xtal_dicts,
    parse_fragalysis_data,
)


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-f",
        "--frag_csv",
        required=True,
        help="Fragalysis crystal structure compound tracker CSV file.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    frag_dir = os.path.dirname(args.frag_csv)

    ## First, parse the fragalysis directory into a dictionary of
    ##  CrystalCompoundData
    sars_xtals = parse_fragalysis_data(args.frag_csv, frag_dir)

    ## Get dict mapping crystal structure id to compound id
    compound_id_dict = get_compound_id_xtal_dicts(sars_xtals.values())[0]

    print(compound_id_dict)

    out_file = "../data/cmpd_to_frag.yaml"
    with open(out_file, "w") as f:
        yaml.safe_dump(compound_id_dict, f)


if __name__ == "__main__":
    main()
