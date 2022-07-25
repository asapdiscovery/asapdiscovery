import argparse
import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from covid_moonshot_ml.docking.docking import build_docking_systems,\
    parse_xtal, run_docking


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-exp', required=True,
                        help='CSV file with experimental data.')
    parser.add_argument('-x', required=True,
                        help='CSV file with crystal compound information.')
    parser.add_argument('-x_dir', required=True,
                        help='Directory with crystal structures.')
    parser.add_argument('-d', help='Directory name to put the structures')
    parser.add_argument('-y', default="mers-structures.yaml",
                        help='MERS structures yaml file')
    parser.add_argument('-r', default=None,
                        help='Path to pdb reference file to align to')
    parser.add_argument('-n', default=None, help='Name of reference')
    return (parser.parse_args())


def main():
    args = get_args()


if __name__ == '__main__':
    main()
