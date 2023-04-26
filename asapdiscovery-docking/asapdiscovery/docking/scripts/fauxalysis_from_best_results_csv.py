"""
Generate fragalysis-like data from a csv file of best results
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.utils import combine_sdf_files


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
    parser.add_argument(
        "--combine_sdfs_only",
        action="store_true",
        help="Only combine sdf files, do not copy protein pdbs",
    )
    parser.add_argument(
        "--move_pdbs_only",
        action="store_true",
        help="Only move protein pdbs, do not combine sdf files",
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
    df = pd.read_csv(args.best_results_csv).replace(np.nan, "None")
    logger.info(f"Loaded {len(df)} rows from {args.best_results_csv}")

    # Get list of sdfs and protein pdbs
    sdf_paths = [Path(path_) for path_ in df["Docked_File"]]
    structure_paths = [Path(path_) for path_ in df["Structure_Path"]]
    dir_names = df["Compound_ID"] + "_" + df["Structure_Source"]
    structure_names = df["Structure_Source"]

    if not len(sdf_paths) == len(structure_paths):
        raise ValueError(
            f"Loaded {len(sdf_paths)} sdf paths and {len(structure_paths)} structure paths, looks like some are missing!"
        )

    # Copy sdfs and protein pdbs to output dir
    # Keep track of which sdf files belong to which structure source
    sdfs_per_structure = {structure_name: [] for structure_name in structure_names}
    for sdf_path, structure_path, dir_name, structure_name in zip(
        sdf_paths, structure_paths, dir_names, structure_names
    ):
        if sdf_path.name == "None":
            logger.error(f"Input csv is missing an sdf path for {dir_name}!")
            continue

        if not sdf_path.exists():
            logger.error(f"{sdf_path} does not exist for {dir_name}!")
            raise FileNotFoundError(f"{sdf_path} does not exist for {dir_name}!")
        if not structure_path.exists() and not args.combine_sdfs_only:
            logger.error(f"{structure_path} does not exist for {dir_name}!")
            raise FileNotFoundError(f"{structure_path} does not exist for {dir_name}!")

        new_dir = args.output_dir / dir_name
        if not new_dir.exists():
            new_dir.mkdir()
        if not args.combine_sdfs_only:
            logger.info(
                f"Copying {sdf_path.name} and {structure_path.name} to {new_dir}"
            )
            shutil.copy2(sdf_path, new_dir)
            shutil.copy2(structure_path, new_dir)
        if not args.move_pdbs_only:
            sdfs_per_structure[structure_name].append(new_dir / sdf_path.name)

    # Combine sdfs into one file
    logger.info(f"Combining sdfs into one per structure source")
    combined_sdf = args.output_dir / "combined.sdf"
    combined_sdf_fd = open(combined_sdf, "wb")
    for structure, paths in sdfs_per_structure.items():
        structure_sdf = args.output_dir / f"{structure}_combined.sdf"
        structure_sdf_fd = (structure_sdf, "wb")
        for sdf_to_copy in paths:
            logger.info(f"Copying {sdf_to_copy} to {structure_sdf}")
            fd = open(sdf_to_copy, "rb")
            shutil.copyfileobj(fd, combined_sdf_fd)
            shutil.copyfileobj(fd, structure_sdf_fd)
            fd.close()
        structure_sdf_fd.close()
    combined_sdf_fd.close()


if __name__ == "__main__":
    main()
