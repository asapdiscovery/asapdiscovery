"""
Input CSV must have the columns:
 * loss_dir
 * label
 * train_frac
"""
import argparse
import logging
import os
import pickle as pkl
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns

# from asapdiscovery.ml.scripts.plot_loss import convert_pic50


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


def load_all_losses(in_df, rel_dir=None, conv_function=None):
    """
    Load all train, val, and test losses, and build DataFrame.

    Parameters
    ----------
    in_df : pandas.DataFrame
        DataFrame containing loss_dir, label, train_frac
    rel_dir : str, optional
        Directory to which all paths in in_df are relative to. If present, will be
        prepended to each path
    conv_function : callable, optional
        If present, will use to convert mean absolute loss values

    Returns
    -------
    pandas.DataFrame
        DataFrame ready with losses, labels, and train fracs
    """
    # Parametrize load_losses function
    load_losses_param = partial(load_losses, conv_function=conv_function)

    all_dirs = [os.path.join(rel_dir, d) if rel_dir else d for d in in_df["loss_dir"]]

    # Load losses and build DF
    df_rows = [load_losses(d) for d in all_dirs]
    df = pandas.DataFrame(df_rows, columns=["train", "val", "test"])

    return pandas.concat([in_df, df], axis=1)


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
        "-i",
        "--in_csv",
        required=True,
        help="CSV file giving directories, labels, and train fracs.",
    )
    parser.add_argument("-o", "--out_fn", required=True, help="Output plot file.")
    parser.add_argument(
        "-d", "--rel_dir", help="Relative directory for all loss_dir in in_csv."
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

    if args.conv:
        # Parametrized function
        convert_pic50_param = partial(
            convert_pic50, T=args.temp, cp_values=args.cheng_prusoff
        )
    else:
        convert_pic50_param = None

    # Load inputs
    in_df = pandas.read_csv(args.in_csv)

    # Build DF from losses
    loss_df = load_all_losses(in_df, args.rel_dir, convert_pic50_param)

    # Prepare for plotting
    plot_df = loss_df.melt(
        id_vars=["label", "train_frac"], var_name="split", value_name="loss"
    )

    # Plot
    plot_data_efficiency(plot_df, args.out_fn)


if __name__ == "__main__":
    main()
