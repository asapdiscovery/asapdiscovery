import sys, os, argparse, shutil

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from asap_docking.docking import DockingResults


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_csv",
        required=True,
        help="Path to CSV file containing best results.",
    )
    parser.add_argument(
        "-f",
        "--fragalysis_dir",
        default=False,
        help="Path to fragalysis results directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to newly created fauxalysis directory",
    )

    return parser.parse_args()


def main():
    args = get_args()

    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    docking_results = DockingResults(args.input_csv)

    for index, values in docking_results.df.to_dict(orient="index").items():
        input_dir_path = os.path.dirname(values["Docked_File"])
        output_dir_path = os.path.join(args.output_dir, values["Complex_ID"])

        if os.path.exists(input_dir_path) and not os.path.exists(
            output_dir_path
        ):
            # print(input_dir_path)
            # print(output_dir_path)
            shutil.copytree(input_dir_path, output_dir_path)

        if args.fragalysis_dir:
            compound_id = values["Compound_ID"]
            compound_path = os.path.join(args.fragalysis_dir, compound_id)
            bound_pdb_path = os.path.join(
                compound_path, f"{compound_id}_bound.pdb"
            )
            if os.path.exists(bound_pdb_path):
                print(bound_pdb_path)
                shutil.copy2(bound_pdb_path, output_dir_path)


if __name__ == "__main__":
    main()
