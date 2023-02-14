import argparse, os, sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
from asapdiscovery.data.pdb import (
    download_PDBs,
    load_pdbs_from_yaml,
)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d",
        "--pdb_dir_path",
        default="../tests/pdb_download",
        help="Directory name to put the structures",
    )
    parser.add_argument(
        "-p",
        "--pdb_yaml_path",
        default="../../../metadata/mers-structures.yaml",
        help="MERS structures yaml file",
    )
    parser.add_argument(
        "-r",
        "--ref_path",
        default=None,
        help="Path to pdb reference file to align to",
    )
    parser.add_argument(
        "-n", "--ref_name", default=None, help="Name of reference"
    )
    parser.add_argument(
        "-t",
        "--pdb_type",
        default="pdb",
        help="pdb_type",
    )
    return parser.parse_args()


def main():
    args = get_args()
    pdb_list = load_pdbs_from_yaml(args.pdb_yaml_path)
    download_PDBs(pdb_list, args.pdb_dir_path, args.pdb_type)


if __name__ == "__main__":
    main()
