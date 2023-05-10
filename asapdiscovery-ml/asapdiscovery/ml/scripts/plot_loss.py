import argparse
import logging
import os
import pickle as pkl
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Compute R value in kcal/mol/K
try:
    from simtk.unit import MOLAR_GAS_CONSTANT_R as R_const
    from simtk.unit import kelvin as K
    from simtk.unit import kilocalorie as kcal
    from simtk.unit import mole as mol

    R = R_const.in_units_of(kcal / mol / K)._value
except ModuleNotFoundError:
    # use R = .001987 kcal/mol/K
    R = 0.001987
    logging.debug("simtk package not found, using R value of", R)


def convert_pic50(pic50, T=298.0, cp_values=None):
    """
    Function to convert pIC50 value to delta G value (in kcal/mol).

    Conversion:
    IC50 value = exp(dG/kT) => pic50 = -log10(exp(dG/kT))
    exp(dg/kT) = 10**(-pic50)
    dg = kT * ln(10**(-pic50))
    change of base => dg = kT * -pic50 / log10(e)

    Parameters
    ----------
    pic50 : float
        pIC50 value to convert
    T : float, default=298.0
        Temperature for conversion
    cp_values : Tuple[int], optional
        Substrate concentration and Km values for calculating Ki using the
        Cheng-Prussoff equation. These values are assumed to be in the same
        concentration units. If no values are passed for this, pIC50 values
        will be used as an approximation of the Ki

    Returns
    -------
    float
        Converted delta G value in kT
    """
    # Calculate Ki using Cheng-Prussoff
    if cp_values:
        # Convert pIC50 -> IC50
        ic50 = 10 ** (-pic50)
        dG = R * T * np.log(ic50 / (1 + cp_values[0] / cp_values[1]))
    # Use Ki = pIC50 approximation
    else:
        dG = -R * T * np.log(10.0) * pic50

    # Plotting MAE so return absolute value
    return np.abs(dG)


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
        "-s",
        "--start_epoch",
        type=int,
        default=0,
        help="Which epoch to start plotting from.",
    )
    parser.add_argument(
        "-e",
        "--end_epoch",
        type=int,
        help="Last epoch to plot.",
    )
    parser.add_argument(
        "-m",
        "--max",
        type=float,
        help="Max error to plot (ie upper bound of loss plot).",
    )

    return parser.parse_args()


def main():
    args = get_args()

    assert len(args.loss_dirs) == len(args.labels), "Incorrect number of labels"

    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Combined lists for plotting
    all_epoch = []
    all_loss = []
    all_lab = []
    all_type = []
    if args.test_only:
        best_loss = {}

    # Parametrized function
    convert_pic50_param = partial(
        convert_pic50, T=args.temp, cp_values=args.cheng_prusoff
    )

    for d, l in zip(args.loss_dirs, args.labels):
        # Load and convert train and test loss
        if not args.test_only:
            train_loss = pkl.load(open(os.path.join(d, "train_err.pkl"), "rb"))
            if args.conv:
                train_loss = np.sqrt(train_loss).mean(axis=1)
                train_loss = convert_pic50_param(train_loss)
            else:
                train_loss = train_loss.mean(axis=1)

            # Add data to the combined lists
            all_epoch.extend(range(len(train_loss)))
            all_loss.extend(train_loss)
            all_lab.extend([l] * len(train_loss))
            all_type.extend(["train"] * len(train_loss))

        test_loss = pkl.load(open(os.path.join(d, "test_err.pkl"), "rb"))
        if args.conv:
            test_loss = np.sqrt(test_loss).mean(axis=1)
            test_loss = convert_pic50_param(test_loss)
        else:
            test_loss = test_loss.mean(axis=1)
        if args.test_only:
            best_loss[l] = test_loss[-1]

        # Add data to the combined lists
        all_epoch.extend(range(len(test_loss)))
        all_loss.extend(test_loss)
        all_lab.extend([l] * len(test_loss))
        all_type.extend(["test"] * len(test_loss))

        # Try to load val loss, but ignore if it's not there
        if not args.test_only:
            try:
                val_loss = pkl.load(open(os.path.join(d, "val_err.pkl"), "rb"))
                if args.conv:
                    val_loss = np.sqrt(val_loss).mean(axis=1)
                    val_loss = convert_pic50_param(val_loss)
                else:
                    val_loss = val_loss.mean(axis=1)

                # Add data to the combined lists
                all_epoch.extend(range(len(val_loss)))
                all_loss.extend(val_loss)
                all_lab.extend([l] * len(val_loss))
                all_type.extend(["val"] * len(val_loss))

            except FileNotFoundError:
                print(f"No val loss file found for {d}", flush=True)
                pass

    all_epoch = np.asarray(all_epoch)
    all_loss = np.asarray(all_loss)
    all_lab = np.asarray(all_lab)
    all_type = np.asarray(all_type)

    idx = all_epoch >= args.start_epoch
    if args.end_epoch is not None:
        idx &= all_epoch <= args.end_epoch
    all_epoch = all_epoch[idx]
    all_loss = all_loss[idx]
    all_lab = all_lab[idx]
    all_type = all_type[idx]

    # Plot
    styles = ["test", "best"] if args.test_only else ["train", "val", "test"]
    sns.lineplot(
        x=all_epoch,
        y=all_loss,
        hue=all_lab,
        style=all_type,
        style_order=styles,
        lw=1,
        estimator=None,
        errorbar=None,
        ax=ax,
        # sort=False
    )

    if args.test_only:
        for (l, min_loss), c in zip(best_loss.items(), sns.color_palette()):
            ax.axhline(y=min_loss, label=f"{l} (min)", color=c, ls="--", lw=1)

    if args.max is not None:
        ax.set_ylim(0, args.max)

    # Fix axes
    ylab = "MAE (delta G in kcal/mol)" if args.conv else "MSE (squared pIC50)"
    ax.set_ylabel(ylab)
    ax.set_xlabel("Epoch")
    title = "delta G MAE Loss" if args.conv else "pIC50 MSE Loss"
    ax.set_title(title)

    # Save plot
    fig.savefig(args.out_fn, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
