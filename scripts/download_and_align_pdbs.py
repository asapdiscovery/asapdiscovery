import argparse, os, yaml, sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
from covid_moonshot_ml.datasets import pdb


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d", "--pdb_dir_path", help="Directory name to put the structures"
    )
    parser.add_argument(
        "-p",
        "--pdb_yaml_path",
        default="../data/mers-structures.yaml",
        help="MERS structures yaml file",
    )
    parser.add_argument(
        "-r", "--ref_path", default=None, help="Path to pdb reference file to align to"
    )
    parser.add_argument("-n", "--ref_name", default=None, help="Name of reference")
    parser.add_argument(
        "-s",
        "--sel_dict_yaml_path",
        default=None,
        help="Path to yaml file containing selection dictionary",
    )
    return parser.parse_args()


def main():
    args = get_args()
    pdb_list = pdb.load_pdbs_from_yaml(args.pdb_yaml_path)
    pdb.download_PDBs(pdb_list, args.pdb_dir_path)
    if args.sel_dict_yaml_path:
        print(f"Loading selection dictionary from {args.sel_dict_yaml_path}...")
        with open(args.sel_dict_yaml_path, "r") as f:
            sel_dict = yaml.safe_load(f)
    else:
        sel_dict = None
    pdb.align_all_pdbs(
        pdb_list,
        args.pdb_dir_path,
        args.ref_path,
        args.ref_name,
        mobile_chain_id="A",
        ref_chain_id="A",
        sel_dict=sel_dict,
    )


if __name__ == "__main__":
    main()
