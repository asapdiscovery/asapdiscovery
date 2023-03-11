import argparse
from asapdiscovery.data.openeye import load_openeye_pdb, save_openeye_pdb


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-i",
        "--input_prot",
        required=True,
        type=str,
        help="Path to pdb file of protein to prep.",
    )
    parser.add_argument(
        "-o",
        "--output_fn",
        required=True,
        type=str,
        help="Path to output_dir.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    mol = load_openeye_pdb(args.input_prot)
    save_openeye_pdb(mol, args.output_fn)


if __name__ == "__main__":
    main()
