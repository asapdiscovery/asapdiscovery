import argparse
import os
import sys
from glob import glob

sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
from asapdiscovery.data.openeye import save_receptor_grid


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_glob",
        required=True,
        type=str,
        help="Glob of design units to load.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Path to output_dir.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(f"{len(glob(args.input_glob))} files found")

    os.makedirs(args.output_dir, exist_ok=True)

    for du_fn in glob(args.input_glob):
        ## get name of directory
        fn_header = os.path.split(os.path.dirname(du_fn))[1]
        out_fn = os.path.join(args.output_dir, f"{fn_header}.ccp4")
        print(f"Writing {du_fn} to {out_fn}")
        save_receptor_grid(du_fn, out_fn)


if __name__ == "__main__":
    main()
