"""
Script to download the COVID Moonshot data from CDD.
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covid_moonshot_ml.datasets.moonshot import download_molecules

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-tok", required=True, help="File containing CDD token."
    )
    parser.add_argument("-o", required=True, help="Output CSV file.")

    return parser.parse_args()


def main():
    args = get_args()

    ## Set up logging
    logging.basicConfig(level=logging.DEBUG)

    header = {"X-CDD-token": "".join(open(args.tok, "r").readlines()).strip()}
    _ = download_molecules(
        header,
        smiles_fieldname="suspected_SMILES",
        retain_achiral=True,
        retain_racemic=True,
        fn_out=args.o,
    )


if __name__ == "__main__":
    main()
