"""
Script to download the COVID Moonshot data from CDD.
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asapdiscovery.data.moonshot import download_molecules

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-tok",
        "--token",
        help=(
            "File containing CDD token. Not used if the CDDTOKEN "
            "environment variable is set."
        ),
    )
    parser.add_argument("-o", required=True, help="Output CSV file.")
    parser.add_argument("-cache", help="Cache CSV file.")

    ## Filtering arguments
    parser.add_argument(
        "-smi",
        "--smiles_fieldname",
        default="suspected_SMILES",
        help="Which column in the downloaded CSV file to use as SMILES.",
    )
    parser.add_argument(
        "--retain_achiral", action="store_true", help="Keep achiral molecules."
    )
    parser.add_argument(
        "--retain_racemic", action="store_true", help="Keep racemic molecules."
    )
    parser.add_argument(
        "--retain_enantiopure",
        action="store_true",
        help="Keep chirally resolved molecules.",
    )
    parser.add_argument(
        "--retain_semiquant",
        action="store_true",
        help="Keep molecules whose IC50 values are out of range.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## Set up logging
    logging.basicConfig(level=logging.DEBUG)

    if "CDDTOKEN" in os.environ:
        header = {"X-CDD-token": os.environ["CDDTOKEN"]}
    elif args.token:
        header = {
            "X-CDD-token": "".join(open(args.token, "r").readlines()).strip()
        }
    else:
        raise ValueError(
            (
                "Must pass a file for -tok if the CDDTOKEN environment "
                "variable is not set."
            )
        )

    _ = download_molecules(
        header,
        smiles_fieldname=args.smiles_fieldname,
        fn_out=args.o,
        fn_cache=args.cache,
        retain_achiral=args.retain_achiral,
        retain_racemic=args.retain_racemic,
        retain_enantiopure=args.retain_enantiopure,
        retain_semiquantitative_data=args.retain_semiquant,
    )


if __name__ == "__main__":
    main()
