import argparse
import json
import os
import pandas
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from covid_moonshot_ml.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-i", required=True, help="Experimental filename.")
    parser.add_argument("-o", required=True, help="Output filename.")

    return parser.parse_args()


def main():
    args = get_args()




if __name__ == "__main__":
    main()
