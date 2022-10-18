import sys, os, argparse

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from asap_docking.docking import DockingDataset


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d", required=True, help="Directory containing docking results."
    )

    parser.add_argument(
        "-f", required=True, help="Directory containing fragalysis data."
    )

    parser.add_argument("-c", required=True, help="Name of output csv file")

    return parser.parse_args()


def main():
    args = get_args()

    pkl_fn = os.path.join(args.d, "mcs_sort_index.pkl")
    dd = DockingDataset(pkl_fn=pkl_fn, dir_path=args.d)
    dd.read_pkl()
    dd.analyze_docking_results(args.f, args.c, test=False)


if __name__ == "__main__":
    main()
