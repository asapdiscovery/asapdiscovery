"""
Script to convert a CSV file downloaded (and filtered) from CDD into Schema
objects that can be used with the rest of the asapdiscovery pipeline. At a
minimum, the CSV file must have the following columns:
 * "smiles" or "suspected_SMILES"
 * "Canonical PostEra ID"
 * "pIC50" or "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50"

Minimal example usage:
python cdd_to_schema.py \
-i cdd_downloaded_filtered.csv \
-json cdd_downloaded_filtered.json
"""

import argparse
import json
import os
import pandas
from rdkit.Chem import MolToSmiles, MolFromSmiles
import re
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from asapdiscovery.data.utils import cdd_to_schema, cdd_to_schema_pair


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-i", required=True, help="CSV file input from CDD.")
    parser.add_argument("-json", required=True, help="Output JSON file.")
    parser.add_argument("-csv", help="Output CSV file.")
    parser.add_argument(
        "-type",
        default="std",
        help=(
            "What type of data is "
            "being loaded (std: standard, ep: enantiomer pairs)"
        ),
    )
    parser.add_argument(
        "-achiral", action="store_true", help="Remove chiral molecules."
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.type.lower() == "std":
        _ = cdd_to_schema(args.i, args.json, args.csv, args.achiral)
    elif args.type.lower() == "ep":
        _ = cdd_to_schema_pair(args.i, args.json, args.csv)
    else:
        raise ValueError(f"Unknown value for -type: {args.type}.")


if __name__ == "__main__":
    main()
