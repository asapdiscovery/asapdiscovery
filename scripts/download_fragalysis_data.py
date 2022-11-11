import argparse
import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from asap_datasets.datasets.fragalysis import download

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-o", required=True, help="Output file name.")
    parser.add_argument(
        "-x", action="store_true", help="Extract file after downloading it."
    )

    return parser.parse_args()


def main():
    args = get_args()

    download(args.o, args.x)


if __name__ == "__main__":
    main()
