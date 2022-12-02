"""
The purpose of this script is to make a multi-ligand SDF file which would be an input to the run_docking_oe.py script.
Currently, the point is to process the fragalysis dataset

structure_dir = "/Users/alexpayne/Scientific_Projects/mers-drug-discovery/mpro-paper-ligand/aligned/"
xtal_csv = "/Users/alexpayne/Scientific_Projects/mers-drug-discovery/mpro-paper-ligand/extra_files/Mpro_compound_tracker_csv.csv"
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-csv",
        "--xtal_csv",
        required=True,
        help="Path to fragalysis Mpro_compound_tracker_csv.csv.",
    )
    parser.add_argument(
        "-o",
        "--sdf_fn",
        required=True,
        help="Path to output multi-object sdf file that will be created",
    )


def main():
    pass


if __name__ == "__main__":
    main()
