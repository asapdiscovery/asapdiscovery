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
        "-tok",
        help=(
            "File containing CDD token. Not used if the CDDTOKEN "
            "environment variable is set."
        ),
    )
    parser.add_argument("-o", required=True, help="Output CSV file.")

    return parser.parse_args()


def main():
    args = get_args()

    ## Set up logging
    logging.basicConfig(level=logging.DEBUG)

    if "CDDTOKEN" in os.environ:
        header = {"X-CDD-token": os.environ["CDDTOKEN"]}
    elif args.tok:
        header = {
            "X-CDD-token": "".join(open(args.tok, "r").readlines()).strip()
        }
    else:
        raise ValueError(
            (
                "Must pass a file for -tok if the CDDTOKEN environment variable "
                "is not set."
            )
        )

    _ = download_molecules(
        header,
        smiles_fieldname="suspected_SMILES",
        retain_achiral=True,
        retain_racemic=True,
        fn_out=args.o,
    )


if __name__ == "__main__":
    main()
