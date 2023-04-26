import argparse
import logging
import os
import pickle as pkl
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    List[float, float, float]
        Train, val, test losses
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

    return [
        loss_arrays["train"][best_idx],
        loss_arrays["val"][best_idx],
        loss_arrays["test"][best_idx],
    ]


def load_all_losses(loss_dirs, labels, train_fracs, conv_function=None):
    """
    Load all train, val, and test losses, and build DataFrame.

    Parameters
    ----------
    loss_dirs : List[str]
        List of directories containing loss pickle files
    labels : List[str]
        List of labels for each dir (one to one)
    train_fracs : List[float]
        List of train fractions for each dir (one to one)
    conv_function : callable, optional
        If present, will use to convert mean absolute loss values

    Returns
    -------
    pandas.DataFrame
        DataFrame ready with losses, labels, and train fracs
    """
    # Parametrize load_losses function
    load_losses_param = partial(load_losses, conv_function=conv_function)

    df_rows = [
        load_losses(d) + [l, train_frac]
        for d, l, train_frac in zip(loss_dirs, labels, train_fracs)
    ]
    return pandas.DataFrame(
        df_rows, columns=["train", "val", "test", "label", "train_frac"]
    )


def plot_data_efficiency(plot_df, out_fn):
    """
    Plot loss as a function of data efficiency. Data points will be grouped by label,
    with each split having a different line style.

    Parameters
    ----------
    plot_df : pandas.DataFrame
        DF ready for plotting. Should have the following columns:
         * label
         * train_frac
         * split
         * loss
    out_fn : str
        File to store plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(plot_df, x="train_frac", y="loss", hue="label", style="split", ax=ax)

    # Set axes
    ax.set_ylabel("Loss")
    ax.set_xlabel("Fraction of Data in Training Split")

    fig.savefig(out_fn, dpi=200, bbox_inches="tight")


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
        type=float,
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
        raise ValueError("Incorrect number of labels.")
    if len(args.loss_dirs) != len(args.train_fracs):
        raise ValueError("Incorrect number of train fracs.")

    if args.conv:
        # Parametrized function
        convert_pic50_param = partial(
            convert_pic50, T=args.temp, cp_values=args.cheng_prusoff
        )
    else:
        convert_pic50_param = None

    # Build DF from losses
    loss_df = load_all_losses(
        args.loss_dirs, args.labels, args.train_fracs, convert_pic50_param
    )

    # Prepare for plotting
    plot_df = loss_df.melt(
        id_vars=["label", "train_frac"], var_name="split", value_name="loss"
    )

    # Plot
    plot_data_efficiency(plot_df, args.out_fn)


if __name__ == "__main__":
    main()
