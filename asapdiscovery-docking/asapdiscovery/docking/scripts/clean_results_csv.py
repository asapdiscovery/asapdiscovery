"""
This script cleans up the CSV currently (as of 2022.10.14) generated by
`hallucinate_fragalysis.py` into a format later scripts can use (such as
fauxalysis_from_docking.py and plot_docking_results.py).

Example Usage:
    python clean_results_csv.py
        -i all_results.csv
        -o posit_hybrid_no_relax_keep_water_filter_frag
        -d
        -s
"""
import argparse
import os

import numpy as np
from asapdiscovery.docking.analysis import DockingResults  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-i",
        "--input_csv",
        required=True,
        help="Path to CSV file containing docking results.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-d",
        "--dimer",
        action="store_true",
        default=False,
        help="If true, keep track of whether the complex is a dimer. Optional as earlier CSVs didn't have this.",  # noqa E501
    )
    parser.add_argument(
        "-s",
        "--clean_structure_source",
        action="store_true",
        default=False,
        help="If true, alter the string in the Structure_Source column to be cleaner",
    )
    parser.add_argument(
        "-r",
        "--resolution_csv",
        default=os.path.join(
            repo_path,
            "data",
            "mers_structures.csv",
        ),
    )
    # TODO: this should be expandable to filter based on multiple different scores and
    # values
    parser.add_argument(
        "-v",
        "--filter_value",
        default=2.5,
        type=float,
        help="Cutoff for filtering docking results when getting the best model.",
    )
    parser.add_argument(
        "-f",
        "--filter_score",
        default="RMSD",
        type=str,
        help="Score to use for filtering docking results when getting the best model.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args.input_csv)
    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    dr = DockingResults(args.input_csv)

    # this is a bunch of csv ingesting that is very dependent on the way the csv looks
    dr.df = dr.df.drop("Unnamed: 0", axis=1)
    if args.clean_structure_source:
        # This drops the unnecessary 'rcsb_' from the structure_source column
        dr.df["MERS_structure"] = dr.df["MERS_structure"].apply(
            lambda x: x.split("_")[1]
        )

    # Rename the columns, add POSIT_R
    print(args.dimer)
    if args.dimer:
        dr.df.columns = [
            "Compound_ID",
            "Structure_Source",
            "Dimer",
            "Docked_File",
            "RMSD",
            "POSIT",
            "Chemgauss4",
            "Clash",
        ]
    else:
        dr.df.columns = [
            "Compound_ID",
            "Structure_Source",
            "Docked_File",
            "RMSD",
            "POSIT",
            "Chemgauss4",
            "Clash",
        ]
    dr.df["POSIT_R"] = 1 - dr.df.POSIT

    # Drop "_bound" from Compound_ID
    # ToDo: deal with this issue of the "_bound" compound IDs in the fragalysis database
    # dr.df.Compound_ID = [
    #     string.replace("_bound", "") for string in dr.df.Compound_ID
    # ]

    # Add Complex_ID
    # This is not the same thing as f"{dr.df.Compound_ID}_{dr.df.Structure_Source}", as
    # this enables rowwise addition
    # as opposed to adding the *entire* series as a single string
    dr.df["Complex_ID"] = dr.df.Compound_ID.apply(str) + "_" + dr.df.Structure_Source

    # Clean the Docked_File paths because there are extra `/`
    # also, some of the file paths are NaNs so we need to only keep the ones that are
    # strings
    dr.df.Docked_File = dr.df.Docked_File.replace(np.nan, "")
    dr.df.Docked_File = [string.replace("//", "/") for string in dr.df.Docked_File]

    # Re-sort the dataframe by the Compound_ID so that its nice and alphabetical and
    # re-index based on that
    dr.df = dr.df.sort_values(["Complex_ID"]).reset_index(drop=True)

    # Get several dataframes
    dr.get_compound_df()
    dr.get_structure_df(resolution_csv=args.resolution_csv)
    dr.get_best_structure_per_compound(
        filter_score=args.filter_score, filter_value=args.filter_value
    )

    # Write out CSV Files
    dr.write_dfs_to_csv(args.output_dir)


if __name__ == "__main__":
    main()
