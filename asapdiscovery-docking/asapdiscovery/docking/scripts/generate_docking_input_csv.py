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
        "--protein_name",
        type=str,
        default=None,
        help="Name of protein.",
    )
    parser.add_argument(
        "--ligand_name",
        type=str,
        default=None,
        help="Name of ligand.",
    )
    parser.add_argument(
        "--protein_regex",
        type=str,
        default=None,
        help="Regex to extract protein name from file name.",
    )
    parser.add_argument(
        "--ligand_regex",
        type=str,
        default=None,
        help="Regex to extract ligand name from file name.",
    )
    parser.add_argument(
        "--split_by_ligand",
        action="store_true",
        default=False,
        help="If true, split out csv by ligand.",
    )
    parser.add_argument(
        "--split_by_protein",
        action="store_true",
        default=False,
        help="If true, split out csv by protein.",
    )
    parser.add_argument(
        "--split_by_n_rows",
        type=int,
        default=None,
        help="Split out csv by number of rows.",
    )
    parser.add_argument(
        "--split_by_n_files",
        type=int,
        default=None,
        help="Split out csv into that many files.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    du_fns = list(glob(args.glob_str))
    rows = []
    for du_fn in du_fns:
        if args.protein_regex:
            protein = re.search(args.protein_regex, du_fn).group(1)
        elif args.protein_name:
            protein = args.protein_name
        else:
            raise ValueError("Must provide either protein_regex or protein_name.")

        if args.ligand_regex:
            ligand = re.search(args.ligand_regex, du_fn).group(1)
        elif args.ligand_name:
            ligand = args.ligand_name
        else:
            raise ValueError("Must provide either ligand_regex or ligand_name.")
        complex_name = f"{protein}_{ligand}"
        rows.append([protein, ligand, complex_name, du_fn])

    if not args.output_csv.parent.exists():
        args.output_csv.parent.mkdir(parents=True)

    df = pd.DataFrame(rows, columns=["protein", "ligand", "complex", "du_fn"])
    df.to_csv(args.output_csv, index=False)

    if args.split_by_ligand:
        for ligand, ligand_df in df.groupby("ligand"):
            ligand_df.to_csv(
                args.output_csv.parent / f"{ligand}_docking_input.csv", index=False
            )
    elif args.split_by_protein:
        for protein, protein_df in df.groupby("protein"):
            protein_df.to_csv(
                args.output_csv.parent / f"{protein}_docking_input.csv", index=False
            )
    elif args.split_by_n_rows:
        import math

        import numpy as np

        for i, chunk in enumerate(
            np.array_split(df, math.ceil(len(df) / args.split_by_n_rows))
        ):
            chunk.to_csv(
                args.output_csv.parent / f"{i+1}_docking_input.csv", index=False
            )
    elif args.split_by_n_files:
        import math

        import numpy as np

        for i, chunk in enumerate(np.array_split(df, args.split_by_n_files)):
            chunk.to_csv(
                args.output_csv.parent / f"{i+1}_docking_input.csv", index=False
            )


if __name__ == "__main__":
    main()
