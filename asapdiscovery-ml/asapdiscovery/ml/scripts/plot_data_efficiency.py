import argparse
from functools import partial
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import seaborn as sns

from asapdiscovery.ml.scripts.plot_loss import convert_pic50


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-d",
        "--loss_dirs",
        required=True,
        nargs="+",
        help="Top-level directories containing train, val, and test losses.",
    )
    parser.add_argument("-o", "--out_fn", required=True, help="Output plot file.")
    parser.add_argument(
        "-l",
        "--labels",
        required=True,
        nargs="+",
        help="List of labels (one for each loss dir).",
    )

    parser.add_argument(
        "--conv",
        action="store_true",
        help="Convert errors from squared pIC50 values to delta G values.",
    )
    parser.add_argument(
        "-T",
        "--temp",
        type=float,
        default=298.0,
        help="Temperature in K to use for delta G conversion.",
    )
    parser.add_argument(
        "-cp",
        "--cheng_prusoff",
        nargs=2,
        type=float,
        default=[0.375, 9.5],
        help=(
            "[S] and Km values to use in the Cheng-Prusoff equation (assumed to be in "
            "the same units). Default values are those used in the SARS-CoV-2 "
            "fluorescence experiments from the COVID Moonshot project (in uM here). "
            "Pass 0 for both values to disable and use the pIC50 approximation."
        ),
    )
    parser.add_argument("--test_only", action="store_true", help="Only plot test loss.")

    parser.add_argument(
        "-m",
        "--max",
        type=float,
        help="Max error to plot (ie upper bound of loss plot).",
    )

    return parser.parse_args()


def main():
    args = get_args()

    if len(args.loss_dirs) != len(args.labels):
        raise ValueError("Incorrect number of labels")


if __name__ == "__main__":
    main()
