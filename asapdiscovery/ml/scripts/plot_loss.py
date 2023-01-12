import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import seaborn as sns


## Set up constants for pIC50 conversion
from simtk.unit import (
    AVOGADRO_CONSTANT_NA,
    BOLTZMANN_CONSTANT_kB as kB,
    calorie,
    coulomb,
    elementary_charge,
    kelvin,
    mole,
    Quantity,
)

# ## Electron volt constant
# eV = elementary_charge.conversion_factor_to(coulomb)
# ## Convert kB to eV (w T=298K)
# kT = (kB / eV * 298.0)._value

## RT for T = 298K (kcal/mol)
RT = (kB * NA * (Quantity(298.0, kelvin))).in_units_of(calorie / mole) / 1000
RT = RT._value


def convert_pic50(pic50):
    """
    Function to convert pIC50 value to delta G value (in kT units).

    Conversion:
    IC50 value = exp(dG/kT) => pic50 = -log10(exp(dg/kT))
    exp(dg/kT) = 10**(-pic50)
    dg = kT * ln(10**(-pic50))
    change of base => dg = kT * -pic50 / log10(e)

    Parameters
    ----------
    pic50 : float
        pIC50 value to convert

    Returns
    -------
    float
        Converted delta G value in kT
    """
    return kT * -pic50 / np.log10(np.e)


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
    parser.add_argument(
        "-o", "--out_fn", required=True, help="Output plot file."
    )
    parser.add_argument(
        "-l",
        "--labels",
        required=True,
        nargs="+",
        help="List of labels (one for each loss dir).",
    )
    # nargs="+",
    # help=(
    #     "List of comma-separated lists of labels. "
    #     "Each comma-separated list must have the same number of entries as "
    #     "the number of directories given."
    # ),

    parser.add_argument(
        "--conv",
        action="store_true",
        help="Convert errors from squared pIC50 values to delta G values.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    assert len(args.loss_dirs) == len(args.labels), "Incorrect number of labels"

    ## Set up figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    ## Combined lists for plotting
    all_epoch = []
    all_loss = []
    all_lab = []
    all_type = []

    for d, l in zip(args.loss_dirs, args.labels):
        ## Load and convert train and test loss
        train_loss = pkl.load(open(os.path.join(d, "train_err.pkl"), "rb"))
        if args.conv:
            train_loss = np.sqrt(train_loss).mean(axis=1)
            train_loss = convert_pic50(train_loss)
        else:
            train_loss = train_loss.mean(axis=1)

        ## Add data to the combined lists
        all_epoch.extend(range(len(train_loss)))
        all_loss.extend(train_loss)
        all_lab.extend([l] * len(train_loss))
        all_type.extend(["train"] * len(train_loss))

        test_loss = pkl.load(open(os.path.join(d, "test_err.pkl"), "rb"))
        if args.conv:
            test_loss = np.sqrt(test_loss).mean(axis=1)
            test_loss = convert_pic50(test_loss)
        else:
            test_loss = test_loss.mean(axis=1)

        ## Add data to the combined lists
        all_epoch.extend(range(len(test_loss)))
        all_loss.extend(test_loss)
        all_lab.extend([l] * len(test_loss))
        all_type.extend(["test"] * len(test_loss))

        ## Try to load val loss, but ignore if it's not there
        try:
            val_loss = pkl.load(open(os.path.join(d, "val_err.pkl"), "rb"))
            if args.conv:
                val_loss = np.sqrt(val_loss).mean(axis=1)
                val_loss = convert_pic50(val_loss)
            else:
                val_loss = val_loss.mean(axis=1)

            ## Add data to the combined lists
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

    idx = all_epoch >= 100
    all_epoch = all_epoch[idx]
    all_loss = all_loss[idx]
    all_lab = all_lab[idx]
    all_type = all_type[idx]

    ## Plot
    sns.lineplot(
        x=all_epoch,
        y=all_loss,
        hue=all_lab,
        style=all_type,
        style_order=["train", "val", "test"],
        lw=1,
        estimator=None,
        errorbar=None,
        # sort=False
    )

    ## Fix axes
    ylab = "MAE (delta G in kT)" if args.conv else "MSE (squared pIC50)"
    ax.set_ylabel(ylab)
    ax.set_xlabel("Epoch")
    title = "delta G MAE Loss" if args.conv else "pIC50 MSE Loss"
    ax.set_title(title)

    ## Save plot
    fig.savefig(args.out_fn, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
