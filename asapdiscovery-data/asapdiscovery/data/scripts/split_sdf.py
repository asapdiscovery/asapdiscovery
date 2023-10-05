"""
The purpose of this script is to split up a multi-ligand SDF file into individual SDF files with integer names in
order to be used in a job array
"""
import argparse
import os
from math import floor

from asapdiscovery.data.openeye import load_openeye_sdfs, save_openeye_sdfs


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
    parser.add_argument(
        "--name_convention",
        choices=["integer", "name"],
        default="integer",
        help="How to name the output files. 'integer' will name them 1.sdf, 2.sdf, etc. 'name' will name them "
        "according to the name of the molecule in the SDF file",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Reading '{args.sdf_fn}'")

    mols = load_openeye_sdfs(args.sdf_fn)
    print(f"Saving {len(mols)} SDF files to '{args.out_dir}'")

    # If the number of molecules is not evenly divisible by the chunk size, we will have a remainder
    # So using the floor function will get the number of evenly sized files we will make
    # And then we can separately save a remainder file
    n_chunks = floor(
        len(mols) / args.chunk_size,
    )
    remainder = len(mols) % args.chunk_size
    print(f"Saving {n_chunks} files of {args.chunk_size} molecules each")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(n_chunks):
        start = i * args.chunk_size
        end = (i + 1) * args.chunk_size

        # List slicing is smart enough to handle out of bounds endpoints
        # So this will work even if end > len(mols), although it shouldn't need to
        mols_chunk = mols[start:end]

        if not len(mols_chunk) == args.chunk_size:
            # I don't know how this could happen but just in case
            raise ValueError(
                f"In trying to slice the molecules into sets of {args.chunk_size}, we have an error:\n"
                f"len(mols[{start}:{end}]) = {len(mols_chunk)} != {args.chunk_size} = args.chunk_size\n"
                f"Did something happen to the molecule list?"
            )
        if args.name_convention == "integer":
            save_openeye_sdfs(mols_chunk, os.path.join(args.out_dir, f"{i+1}.sdf"))
        elif args.name_convention == "name":
            save_openeye_sdfs(
                mols_chunk,
                os.path.join(args.out_dir, f"{mols_chunk[0].GetTitle()}.sdf"),
            )

    if remainder:
        print(f"Saving {remainder} remainder molecules to {n_chunks+1}.sdf")
        mols_chunk = mols[-remainder:]
        save_openeye_sdfs(mols_chunk, os.path.join(args.out_dir, f"{n_chunks+1}.sdf"))


if __name__ == "__main__":
    main()
