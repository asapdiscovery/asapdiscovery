"""
Generate fragalysis-like data from a csv file of best results
"""
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from asapdiscovery.data.logging import FileLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--best_results_csv",
        required=True,
        type=Path,
        help="Path to CSV file containing best results from docking",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=Path,
        help="Directory to save output files to",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    logger = FileLogger("fauxalysis_from_best_results_csv", args.output_dir).getLogger()
    logger.info(f"Loading files from {args.best_results_csv}")

    # Load CSV
    df = pd.read_csv(args.best_results_csv).replace(np.nan, "")
    logger.info(f"Loaded {len(df)} rows from {args.best_results_csv}")

    # Get list of sdfs and protein pdbs
    sdf_paths = [Path(path_) for path_ in df["Docked_File"] if Path(path_).exists()]
    structure_paths = [
        Path(path_) for path_ in df["Structure_Path"] if Path(path_).exists()
    ]
    dir_names = df["Compound_ID"] + "_" + df["Structure_Source"]

    logger.info(
        f"Found {len(sdf_paths)} sdf files and {len(structure_paths)} protein pdbs"
    )


if __name__ == "__main__":
    main()
