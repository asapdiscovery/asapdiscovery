"""

"""

import re
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import pandas as pd


def get_args():
    parser = ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-g",
        "--glob_str",
        required=True,
        type=str,
        help="Path/glob to prepped receptor(s)",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        required=True,
        type=Path,
        help="Path to output csv file.",
    )
    parser.add_argument(
        "--protein_regex",
        type=str,
        help="Regex to extract protein name from file name.",
    )
    parser.add_argument(
        "--ligand_regex",
        type=str,
        help="Regex to extract ligand name from file name.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    du_fns = list(glob(args.glob_str))
    rows = []
    for du_fn in du_fns:
        protein = re.search(args.protein_regex, du_fn).group(1)
        ligand = re.search(args.ligand_regex, du_fn).group(1)
        complex_name = f"{protein}_{ligand}"
        rows.append([protein, ligand, complex_name, du_fn])
    df = pd.DataFrame(rows, columns=["protein", "ligand", "complex", "du_fn"])
    if not args.output_csv.parent.exists():
        args.output_csv.parent.mkdir(parents=True)
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
