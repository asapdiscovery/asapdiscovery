import sys, os, argparse
import numpy as np

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from asap_docking import DockingResults


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_csv",
        required=True,
        help="Path to CSV file containing docking results.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output directory"
    )

    return parser.parse_args()


def main():
    args = get_args()

    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    dr = DockingResults(args.input_csv)

    ## this is a bunch of csv ingesting that is very dependent on the way the csv looks
    dr.df = dr.df.drop("Unnamed: 0", axis=1)
    dr.df["MERS_structure"] = dr.df["MERS_structure"].apply(
        lambda x: x.split("_")[1]
    )

    ## Rename the columns, add POSIT_R
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

    ## Drop "_bound" from Compound_ID
    dr.df.Compound_ID = [
        string.replace("_bound", "") for string in dr.df.Compound_ID
    ]

    ## Add Complex_ID
    dr.df["Complex_ID"] = f"{dr.df.Compound_ID}_{dr.df.Structure_Source}"

    ## Clean the Docked_File paths because there are extra `/`
    ## also, some of the file paths are NaNs so we need to only keep the ones that are strings
    dr.df.Docked_File = dr.df.Docked_File.replace(np.nan, "")
    dr.df.Docked_File = [
        string.replace("//", "/") for string in dr.df.Docked_File
    ]

    ## Re-sort the dataframe by the Compound_ID so that its nice and alphabetical and re-index based on that
    dr.df = dr.df.sort_values(["Compound_ID"]).reset_index(drop=True)

    ## Get several dataframes
    dr.get_compound_df()
    dr.get_structure_df(resolution_csv="../data/mers_structures.csv")
    dr.get_best_structure_per_compound()

    ## Write out CSV Files
    dr.df.to_csv(
        os.path.join(args.output_dir, "all_results_cleaned.csv"), index=False
    )
    dr.compound_df.to_csv(
        os.path.join(args.output_dir, "by_compound.csv"), index=False
    )
    dr.structure_df.to_csv(
        os.path.join(args.output_dir, "by_structure.csv"), index=False
    )
    dr.best_df.to_csv(
        os.path.join(args.output_dir, "mers_fauxalysis.csv"), index=False
    )


if __name__ == "__main__":
    main()
