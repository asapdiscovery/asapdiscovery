import sys, os, argparse, yaml, shutil

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)

from kinoml.databases.pdb import download_pdb_structure


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d", "--fauxalysis_dir", required=True, help="Path to fauxalysis_dir."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output dir"
    )
    parser.add_argument(
        "-y",
        "--yaml_file",
        default="../data/luttens2022ultralarge.yaml",
        help="Path to yaml_file",
    )

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.yaml_file, "r") as f:
        pdb_dict = yaml.safe_load(f)

    complex_list = [
        directory
        for directory in os.listdir(args.fauxalysis_dir)
        if not ".csv" in directory
    ]

    for file_name in os.listdir(args.fauxalysis_dir):
        if "csv" in file_name or "sdf" in file_name:
            in_file = os.path.join(args.fauxalysis_dir, file_name)
            shutil.copy2(in_file, args.output_dir)

    for pdb in pdb_dict.keys():
        complexes = [
            complex_id
            for complex_id in complex_list
            if pdb.lower() in complex_id
        ]
        for complex in complexes:
            in_path = os.path.join(args.fauxalysis_dir, complex)
            out_path = os.path.join(args.output_dir, complex)
            try:
                shutil.copytree(in_path, out_path)
                print(
                    f"Copying...\n" f"\tfrom: {in_path}" f"\t  to: {out_path}"
                )
            except FileExistsError:
                print(
                    f"File exists, skipping!\n"
                    f"\tfrom: {in_path}"
                    f"\t  to: {out_path}"
                )
                pass


if __name__ == "__main__":
    main()
