import sys, os, argparse, yaml

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
        "-y",
        "--yaml_file",
        default="../data/luttens2022ultralarge.yaml",
        help="Path to yaml_file",
    )
    parser.add_argument()

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

    for pdb in pdb_dict.keys():
        complexes = [
            complex_id
            for complex_id in complex_list
            if pdb.lower() in complex_id
        ]

        for complex in complexes:
            path = os.path.join(args.fauxalysis_dir, complex)

            print(f"Downloading {pdb} to {path}...")
            download_pdb_structure(pdb, path)


if __name__ == "__main__":
    main()
