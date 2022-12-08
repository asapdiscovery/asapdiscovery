"""
Small script to gather all completed results.
"""
import argparse
from glob import glob
import pandas
import pickle as pkl
import shutil

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-r",
        "--results",
        required=True,
        help="Path to all results pickle files.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        help="Directory to write output files.",
    )

    parser.add_argument(
        "--no_sdf", action="store_true", help="Don't write the SDF file."
    )
    parser.add_argument(
        "--no_res",
        action="store_true",
        help="Don't write the results CSV file.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    all_pickle = glob(args.results)

    ## Build final df
    results_cols = [
        "ligand_id",
        "du_structure",
        "docked_file",
        "docked_RMSD",
        "POSIT_prob",
        "POSIT_method",
        "chemgauss4_score",
        "clash",
        "SMILES",
    ]
    results_df = [pkl.load(open(fn, "rb")) for fn in all_pickle]
    results_df = pandas.DataFrame(results_df, columns=results_cols)

    if not args.no_res:
        results_df.to_csv(f"{args.out_dir}/all_results.csv")

    if not args.no_sdf:
        ## Concatenate all individual SDF files
        combined_sdf = f"{args.out_dir}/combined.sdf"
        with open(combined_sdf, "wb") as wfd:
            for f in results_df["docked_file"]:
                if f == "":
                    continue
                with open(f, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)


if __name__ == "__main__":
    main()
