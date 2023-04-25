import argparse
from functools import partial
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import seaborn as sns
from pathlib import Path

from asapdiscovery.ml.scripts.plot_loss import convert_pic50


def load_losses(loss_dir, conv_function=None):
    """
    Load train, val, and test losses from `loss_dir`, converting as necessary. Take loss
    from epoch with best original val loss.

    Parameters
    ----------
    loss_dir : str
        Directory containing pickle files
    conv_function : callable, optional
        If present, will use to convert mean absolute loss values

    Returns
    -------
    float
        Train loss
    float
        Val loss
    float
        Test loss
    """
    p = Path(loss_dir)
    loss_arrays = {}
    for sp in ["train", "val", "test"]:
        fn = p / f"{sp}_err.pkl"
        try:
            loss_arrays[sp] = pkl.load(fn.open("rb"))
        except FileNotFoundError:
            raise FileNotFoundError(f"No {sp} loss found for {loss_dir}.")

    best_idx = np.argmin(loss_arrays["val"].mean(axis=1))

    if conv_function:
        for sp, loss_arr in loss_arrays.items():
            # First convert from squared loss to mean abs loss
            tmp_loss = np.sqrt(loss_arr).mean(axis=1)
            # Then convert from pIC50 to dG
            tmp_loss = conv_function(tmp_loss)
            # Store back into dict
            loss_arrays[sp] = tmp_loss
    else:
        # Just take mean
        loss_arrays = {
            sp: loss_arr.mean(axis=1) for sp, loss_arr in loss_arrays.items()
        }

    return (
        loss_arrays["train"][best_idx],
        loss_arrays["val"][best_idx],
        loss_arrays["test"][best_idx],
    )


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
        "-t",
        "--train_fracs",
        required=True,
        nargs="+",
        help="List of training fractions (one for each loss dir).",
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
