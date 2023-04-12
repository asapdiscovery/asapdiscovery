"""
The purpose of this script is to split up a multi-ligand SDF file into individual SDF files with integer names in 
order to be used in a job array
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-i",
        "--sdf_fn",
        required=True,
        help="Path to input multi-object sdf file that will be split up",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        help="Path to output directory where the individual sdf files will be saved",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        default=1,
        type=int,
        help="Number of molecules to save in each SDF file",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Reading '{args.sdf_fn}'")
    from asapdiscovery.data.openeye import load_openeye_sdfs

    mols = load_openeye_sdfs(args.sdf_fn)
    print(f"Saving {len(mols)} SDF files to '{args.out_dir}'")
    from asapdiscovery.data.openeye import save_openeye_sdfs

    from math import ceil

    n_chunks = ceil(
        len(mols) / args.chunk_size,
    )
    remainder = len(mols) % args.chunk_size

    print(f"Saving {n_chunks} files of {args.chunk_size} molecules each")
    if not remainder == 0:
        print(f"Saving {remainder} molecules in the last file")

    import os

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(n_chunks):
        start = i * args.chunk_size
        end = (i + 1) * args.chunk_size
        mols_chunk = mols[start:end]
        save_openeye_sdfs(mols_chunk, os.path.join(args.out_dir, f"{i+1}.sdf"))


if __name__ == "__main__":
    main()
