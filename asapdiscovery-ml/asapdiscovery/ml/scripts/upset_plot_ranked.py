import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas
# # TODO: Do we need to add upsetplot to our environment yaml?
import upsetplot as ups


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        required=True,
        nargs="+",
        help="pkl files containing lists of correct and incorrect pairs.",
    )
    parser.add_argument(
        "-labs", nargs="+", help="Plot labels (must align with args to -i)."
    )
    parser.add_argument("-o", required=True, help="Output plot filename.")

    return parser.parse_args()


def main():
    args = get_args()

    if args.labs is None or len(args.labs) != len(args.i):
        labs = [os.path.basename(fn)[:-4] for fn in args.i]
    else:
        labs = args.labs

    correct_dict, incorrect_dict = zip(*[pkl.load(open(fn, "rb")) for fn in args.i])
    correct_dict = {lab: l for lab, l in zip(labs, correct_dict)}
    incorrect_dict = {lab: l for lab, l in zip(labs, incorrect_dict)}

    all_pairs = {p for l in correct_dict.values() for p in l}
    all_pairs.update({p for l in incorrect_dict.values() for p in l})

    ## Plotting
    membership_dict = {p: [l for l in labs if p in correct_dict[l]] for p in all_pairs}
    fig = plt.figure(figsize=(12, 8))
    ups.UpSet(ups.from_memberships(membership_dict.values()), subset_size="count").plot(
        fig
    )
    fig.savefig(f"{args.o}.png", dpi=200, bbox_inches="tight")

    ## Formatting to CSV
    membership_arr = np.vstack(
        [[p in correct_dict[l] for l in labs] for p in all_pairs]
    )
    idx = pandas.MultiIndex.from_tuples(all_pairs, names=["active", "inactive"])
    df = pandas.DataFrame(membership_arr, index=idx, columns=labs)
    df = df.sort_values(by=labs, ascending=False)
    df.to_csv(f"{args.o}.csv")


if __name__ == "__main__":
    main()
