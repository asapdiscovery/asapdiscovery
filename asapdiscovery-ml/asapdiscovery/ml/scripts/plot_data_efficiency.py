"""
Input CSV must have the columns:
 * loss_dir
 * label
 * train_frac
"""

import argparse
import json
import os
import pickle as pkl
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from asapdiscovery.ml.scripts.plot_loss import convert_pic50


def load_losses(loss_dir, conv_function=None, agg=np.mean):
    """
    Load train, val, and test losses from `loss_dir`, converting as necessary. Take loss
    from epoch with best original val loss.

    Parameters
    ----------
    loss_dir : str
        Directory containing pickle files
    conv_function : callable, optional
        If present, will use to convert mean absolute loss values
    agg : callable, default=mean
        Function to aggregate loss across molecules. Set to None to return loss for all
        molecules


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

    # Extract losses for best_idx
    loss_arrays = {sp: loss_arr[best_idx, :] for sp, loss_arr in loss_arrays.items()}

    if conv_function:
        for sp, loss_arr in loss_arrays.items():
            # First convert from squared loss to mean abs loss
            tmp_loss = np.sqrt(loss_arr)
            # Then convert from pIC50 to dG
            tmp_loss = conv_function(tmp_loss)
            # Store back into dict
            loss_arrays[sp] = tmp_loss

    # Aggregate
    if agg:
        loss_arrays = {sp: [agg(loss_arr)] for sp, loss_arr in loss_arrays.items()}

    return (loss_arrays["train"], loss_arrays["val"], loss_arrays["test"])


def load_all_losses(in_df, rel_dir=None, conv_function=None, agg=np.mean):
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
    agg : callable, default=mean
        Function to aggregate loss across molecules. Set to None to return loss for all
        molecules


    Returns
    -------
    pandas.DataFrame
        DataFrame ready with losses, labels, and train fracs
    """
    # Parametrize load_losses function
    load_losses_param = partial(load_losses, conv_function=conv_function, agg=agg)

    all_dfs = []
    for _, r in in_df.iterrows():
        # Set up directory
        d = r["loss_dir"]
        if rel_dir:
            d = os.path.join(rel_dir, d)

        # Build melted loss DF
        losses = load_losses_param(d)
        df = losses_to_df(losses)

        df["label"] = r["label"]
        df["train_frac"] = r["train_frac"]
        df["loss_dir"] = r["loss_dir"]

        all_dfs.append(df)

    # all_dirs = [os.path.join(rel_dir, d) if rel_dir else d for d in in_df["loss_dir"]]

    # # Load losses and build DF
    # df_rows = [load_losses_param(d) for d in all_dirs]
    # df = pandas.DataFrame(df_rows, columns=["train", "val", "test"])

    return pandas.concat(all_dfs, axis=0, ignore_index=True)


def losses_to_df(losses):
    """
    Convert a list of lists of losses into a DF with each loss labeled by split.

    Parameters
    ----------
    losses : List[np.ndarray]
        List of losses

    Returns
    -------
    pandas.DataFrame
        Loss DF
    """
    all_labs = []
    all_losses = []
    for lab, loss_arr in zip(["train", "val", "test"], losses):
        all_losses.extend(loss_arr)
        all_labs.extend([lab] * len(loss_arr))

    return pandas.DataFrame({"split": all_labs, "loss": all_losses})


def load_losses_dict(loss_dir, conv_function=None, take_best=True):
    """
    Load train, val, and test losses from `loss_dir`/loss_dict.json, converting as
    necessary. Take loss from epoch with best original val loss.

    Parameters
    ----------
    loss_dir : str
        Directory containing pickle files
    conv_function : callable, optional
        If present, will use to convert mean absolute loss values

    Returns
    -------
    pandas.DataFrame
        DF with losses for the best idx across all splits
    """
    try:
        loss_dict = json.load(open(f"{loss_dir}/loss_dict.json"))
    except FileNotFoundError:
        raise FileNotFoundError(f"No loss_dict.json file found in {loss_dir}")

    # Load per-epoch losses for each molecule, and transpose so each row is an epoch and
    #  each col is a molecule
    all_losses = np.asarray(
        [v["losses"] for d in loss_dict.values() for v in d.values()]
    ).T
    all_compound_ids = np.asarray(
        [compound_id for d in loss_dict.values() for compound_id in d]
    )
    split_label_dict = {
        compound_id: sp for sp, d in loss_dict.items() for compound_id in d
    }

    # Build and rearrange DF
    df = pandas.DataFrame(all_losses, columns=all_compound_ids)
    df = df.reset_index().melt(
        id_vars=["index"], var_name="compound_id", value_name="loss"
    )
    df["split"] = [split_label_dict[c] for c in df["compound_id"]]
    df = df.rename(columns={"index": "epoch"})

    # Take only best idx
    if take_best:
        # Get best epoch idx from val set
        val_arr = np.asarray([v["losses"] for v in loss_dict["val"].values()]).T
        best_idx = np.argmin(val_arr.mean(axis=1))
        df = df.loc[df["epoch"] == best_idx, :]

    # Convert losses if desired
    if conv_function:
        # Stored loss values are in squared pIC50 units, so take sqrt to get plain pIC50
        df.loc[:, "loss"] = df["loss"].pow(0.5).apply(conv_function)

    return df


def load_all_losses_dict(in_df, rel_dir=None, conv_function=None, take_best=True):
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
    load_losses_param = partial(
        load_losses_dict, conv_function=conv_function, take_best=take_best
    )

    all_dfs = []
    for _, r in in_df.iterrows():
        # Set up directory
        d = r["loss_dir"]
        if rel_dir:
            d = os.path.join(rel_dir, d)

        # Build melted loss DF
        df = load_losses_param(d)

        for c in r.index:
            df[c] = r[c]

        all_dfs.append(df)

    return pandas.concat(all_dfs, axis=0, ignore_index=True)


def plot_data_efficiency(plot_df, out_fn, max_loss=None, conv=False):
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

    # Set upper y limit
    if max_loss:
        ax.set_ylim(0, max_loss)

    # Set axes
    ylab = r"MAE ($\Delta$G in kcal/mol)" if conv else "MSE (squared pIC50)"
    ax.set_ylabel(ylab)
    ax.set_xlabel("Fraction of Data in Training Split")
    title = r"$\Delta$G MAE Loss" if conv else "pIC50 MSE Loss"
    ax.set_title(title)

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
    try:
        loss_df = load_all_losses(in_df, args.rel_dir, convert_pic50_param)
    except FileNotFoundError:
        loss_df = load_all_losses_dict(in_df, args.rel_dir, convert_pic50_param)

    # Get rid of other cols if we only want test
    if args.test_only:
        loss_df = loss_df.drop(columns=["train", "val"])

    # Prepare for plotting
    plot_df = loss_df.melt(
        id_vars=["loss_dir", "label", "train_frac"], var_name="split", value_name="loss"
    )

    # Plot
    plot_data_efficiency(plot_df, args.out_fn, args.max, args.conv)


if __name__ == "__main__":
    main()
