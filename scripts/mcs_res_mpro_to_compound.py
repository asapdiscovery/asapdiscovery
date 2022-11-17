"""
Convert Mpro dataset in MCSS results to compound id of docked compound.
"""
import argparse
import pickle as pkl
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covid_moonshot_ml.datasets.utils import get_compound_id_xtal_dicts
from covid_moonshot_ml.docking import parse_xtal

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--in_file", required=True, help="Input pickle file."
    )
    parser.add_argument(
        "-o", "--out_file", required=True, help="Output pickle file."
    )
    parser.add_argument(
        "-x", "--xtal_file", required=True, help="Structure compound CSV file."
    )
    parser.add_argument(
        "-d", "--xtal_dir", required=True, help="Structure directory."
    )

    return parser.parse_args()


def main():
    args = get_args()

    compound_ids, xtal_ids, sort_idxs = pkl.load(open(args.in_file, "rb"))

    ## Parse crystal structures
    xtal_compounds = parse_xtal(args.xtal_file, args.xtal_dir)

    ## Get dict mapping from mpro dataset to compound_id
    xtal_to_compound = get_compound_id_xtal_dicts(xtal_compounds)[1]

    ## Map xtal ids to compound ids
    xtal_compound_ids = list(map(xtal_to_compound.get, xtal_ids))

    pkl.dump(
        [compound_ids, xtal_compound_ids, sort_idxs], open(args.out_file, "wb")
    )


if __name__ == "__main__":
    main()
