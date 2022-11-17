import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas
import seaborn as sns
import torch

from train import init


def predict(model, structure, model_call=lambda model, d: model(d)):
    with torch.no_grad():
        pred = model_call(model, structure)

    return pred.item()


def model_call_schnet(model, d):
    return model(d["z"], d["pos"])


def model_call_schnet_dg(model, d):
    return model(d["z"], d["pos"], d["lig"])


def model_call_e3nn(model, d):
    return model(d)


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i", required=True, help="Input directory containing docked PDB files."
    )
    parser.add_argument(
        "-exp", required=True, help="JSON file giving experimental results."
    )
    parser.add_argument("-wts", help="Model weights file.")
    parser.add_argument("-model_params", help="e3nn model parameters.")
    parser.add_argument("-qm9", help="QM9 directory for pretrained model.")
    parser.add_argument(
        "-qm9_target", type=int, default=10, help="QM9 pretrained target."
    )

    ## Output arguments
    parser.add_argument("-o", required=True, help="Output file basename.")

    ## Model parameters
    parser.add_argument(
        "-model",
        required=True,
        help="Which type of model to use (e3nn or schnet).",
    )
    parser.add_argument(
        "-lig",
        action="store_true",
        help="Whether to treat the ligand and protein atoms separately.",
    )
    parser.add_argument(
        "-dg",
        action="store_true",
        help="Whether to predict pIC50 directly or via dG prediction.",
    )
    parser.add_argument(
        "-rm_atomref",
        action="store_true",
        help="Remove atomref embedding in QM9 pretrained SchNet.",
    )

    ## Eval arguments
    parser.add_argument(
        "-n", type=int, default=12, help="Number of concurrent processes."
    )
    # parser.add_argument('-device', default='cuda',
    #     help='Device to use for training (defaults to GPU).')

    return parser.parse_args()


def main():
    args = get_args()
    exp_affinities, ds_train, ds_test, model, _ = init(args)
    full_ds = ds_train + ds_test

    ## Have to use actual functions bc lambda functions aren't pickleable
    ##  so can't pass them as args in multiprocessing
    if args.model == "e3nn":
        model_call = model_call_e3nn
    elif args.model == "schnet":
        if args.dg:
            model_call = model_call_schnet_dg
        else:
            model_call = model_call_schnet
    else:
        ## Should already be checked in init() but just in case
        raise ValueError(f"Unknown model type {args.model}.")

    if args.wts is not None:
        ## Model was trained on GPU so need to let torch know we're loading for CPU
        model.load_state_dict(
            torch.load(args.wts, map_location=torch.device("cpu"))
        )

    ## No learning going on so we can multiprocess the predictions
    mp_args = [(model, s, model_call) for _, s in full_ds]
    n_procs = min(mp.cpu_count(), len(full_ds), args.n)
    with mp.Pool(n_procs) as pool:
        predictions = pool.starmap(predict, mp_args)

    ## Compile info and save CSV file
    compound_ids = [compound_id for (_, compound_id), _ in full_ds]
    ## Experimentally found pIC50 vals
    target_vals = [exp_affinities[c] for c in compound_ids]
    compare_df = pandas.DataFrame(
        {
            "target": target_vals,
            "pred": predictions,
            "train_test": ["Train"] * len(ds_train) + ["Test"] * len(ds_test),
        },
        index=compound_ids,
    )
    compare_df.to_csv(f"{args.o}.csv")

    ## Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="target",
        y="pred",
        hue="train_test",
        data=compare_df,
        alpha=0.7,
        ax=ax,
    )
    sns.lineplot(
        x=[min(target_vals), max(target_vals)],
        y=[min(target_vals), max(target_vals)],
        ls="--",
        color="black",
        ax=ax,
    )
    fig.savefig(f"{args.o}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
