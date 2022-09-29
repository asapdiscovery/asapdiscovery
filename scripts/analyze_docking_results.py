import sys, os, argparse
import numpy as np

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from covid_moonshot_ml.docking.analysis import DockingResults


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

    ## Get several dataframes
    dr.get_compound_df(score_columns=["RMSD", "POSIT", "Chemgauss4", "POSIT_R"])
    dr.get_structure_df(
        # resolution_csv="../data/mers_structures.csv",
        score_columns=["RMSD", "POSIT", "Chemgauss4", "POSIT_R"],
    )
    dr.get_best_structure_per_compound(
        score_order=["POSIT", "Chemgauss4", "RMSD"],
        score_ascending=[False, True, True],
    )

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
