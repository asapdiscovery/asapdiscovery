"""
Attempt to train a ligand-only graph network using the same functions as the
structure-based models. Use a bunch of stuff from dgl-lifesci.
"""
import argparse
from dgllife.data import MoleculeCSVDataset
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    RandomSplitter,
    SMILESToBigraph,
)
from glob import glob
import json
import numpy as np
import os
import pandas
import pickle as pkl
import re
import sys
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covid_moonshot_ml.data.dataset import GraphDataset
from covid_moonshot_ml.nn import GAT
from covid_moonshot_ml.schema import ExperimentalCompoundDataUpdate
from covid_moonshot_ml.utils import plot_loss


def train(
    model, optimizer, loss_func, ds_train, ds_val, ds_test, n_epochs, device
):
    """
    Training function based on `covid_moonshot_ml.utils.train`.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    optimizer : torch.optim.Optimizer
        Model optimizer
    loss_func : torch.nn.Module
        Loss functions
    ds_train : data.dataset.DockedDataset
        Train dataset to train on
    ds_val : data.dataset.DockedDataset
        Validation dataset to evaluate on
    ds_test : data.dataset.DockedDataset
        Test dataset to evaluate on
    n_epochs : int
        Number of epochs to train for
    device : torch.device
        Where to run the training

    Returns
    -------
    torch.nn.Module
        Trained model
    numpy.ndarray
        Loss for each structure in `ds_train` from each epoch of training, with
        shape (`n_epochs`, `len(ds_train)`)
    numpy.ndarray
        Loss for each structure in `ds_val` from each epoch of training, with
        shape (`n_epochs`, `len(ds_val)`)
    numpy.ndarray
        Loss for each structure in `ds_test` from each epoch of training, with
        shape (`n_epochs`, `len(ds_test)`)
    """
    ## Initialize lists to keep track of loss over time
    train_loss = []
    val_loss = []
    test_loss = []

    ## Send model to device (in case it's not already there)
    model = model.to(device)

    ## Main loop
    for epoch_idx in range(n_epochs):
        print(f"Epoch {epoch_idx}/{n_epochs}", flush=True)
        if epoch_idx % 10 == 0 and epoch_idx > 0:
            print(f"Training error: {np.mean(train_loss[-1]):0.5f}")
            print(f"Validation error: {np.mean(val_loss[-1]):0.5f}")
            print(f"Testing error: {np.mean(test_loss[-1]):0.5f}", flush=True)
        ## Train model
        tmp_loss = []
        for i, (_, g, label, _) in enumerate(ds_train):
            g = g.to(device)
            node_feats = g.ndata["h"].to(device)
            pred = model(g, node_feats)
            optimizer.zero_grad()
            loss = loss_func(pred, label.to(device))
            tmp_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss.append(tmp_loss)
        epoch_train_loss = np.mean(tmp_loss)

        ## Evaluate model
        with torch.no_grad():
            tmp_loss = []
            for _, g, label, _ in ds_val:
                g = g.to(device)
                node_feats = g.ndata["h"].to(device)
                pred = model(g, node_feats)
                loss = loss_func(pred, label.to(device))
                tmp_loss.append(loss.item())
            val_loss.append(tmp_loss)
            epoch_val_loss = np.mean(tmp_loss)

            tmp_loss = []
            for _, g, label, _ in ds_test:
                g = g.to(device)
                node_feats = g.ndata["h"].to(device)
                pred = model(g, node_feats)
                loss = loss_func(pred, label.to(device))
                tmp_loss.append(loss.item())
            test_loss.append(tmp_loss)
            epoch_test_loss = np.mean(tmp_loss)

        wandb.log(
            {
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "test_loss": epoch_test_loss,
                "epoch": epoch_idx,
            }
        )
    return (
        model,
        np.vstack(train_loss),
        np.vstack(val_loss),
        np.vstack(test_loss),
    )


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

    ## Output arguments
    parser.add_argument("-model_o", help="Where to save model weights.")
    parser.add_argument("-plot_o", help="Where to save training loss plot.")
    parser.add_argument("-cache", help="Cache directory for dataset.")

    ## Model arguments
    parser.add_argument(
        "-config",
        required=True,
        help="JSON file containing model config options.",
    )

    ## Performance arguments
    parser.add_argument(
        "-n_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train for.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## Build dataframe for constructing dataset
    ## Get all docked structures
    all_fns = glob(f"{args.i}/*complex.pdb")
    ## Extract crystal structure and compound id from file name
    re_pat = (
        r"(Mpro-.*?_[0-9][A-Z]).*?([A-Z]{3}-[A-Z]{3}-[0-9a-z]{8}-[0-9]+)"
        "_complex.pdb"
    )
    matches = [re.search(re_pat, fn) for fn in all_fns]
    xtal_compounds = [m.groups() for m in matches if m]
    num_found = len(xtal_compounds)

    ## Load the experimental compounds
    exp_compounds = ExperimentalCompoundDataUpdate(
        **json.load(open(args.exp, "r"))
    ).compounds
    ## Filter out molecules with no pIC50 values
    exp_compounds = [
        c for c in exp_compounds if ("pIC50" in c.experimental_data)
    ]

    ## Get compounds that have both structure and experimental data (this step
    ##  isn't actually necessary for performance, but allows a more fair
    ##  comparison between 2D and 3D models)
    xtal_compound_ids = {c[1] for c in xtal_compounds}

    ## Trim exp_compounds
    exp_compounds = [
        c for c in exp_compounds if c.compound_id in xtal_compound_ids
    ]

    ## Dictionary mapping from compound_id to Mpro dataset
    compound_id_dict = {c[1]: c[0] for c in xtal_compounds}

    ## Make cache directory as necessary
    if args.cache is None:
        cache_dir = os.path.join(args.model_o, ".cache")
    else:
        cache_dir = args.cache
    os.makedirs(cache_dir, exist_ok=True)

    ## Build dataset
    ds = GraphDataset(
        exp_compounds,
        compound_id_dict,
        node_featurizer=CanonicalAtomFeaturizer(),
        cache_file=os.path.join(cache_dir, "graph.bin"),
    )

    ## Build dataframe
    all_compound_ids, all_smiles, all_pic50 = zip(
        *[
            (c.compound_id, c.smiles, c.experimental_data["pIC50"])
            for c in exp_compounds
        ]
    )

    ## Split dataset into train/val/test (80/10/10 split)
    n_train = int(len(ds) * 0.8)
    n_val = int(len(ds) * 0.1)
    n_test = len(ds) - n_train - n_val
    print(
        (
            f"{n_train} training samples, {n_val} validation samples, "
            f"{n_test} testing samples"
        ),
        flush=True,
    )
    # use fixed seed for reproducibility
    ds_train, ds_val, ds_test = torch.utils.data.random_split(
        ds, [n_train, n_val, n_test], torch.Generator().manual_seed(42)
    )

    exp_configure = json.load(open(args.config))
    exp_configure.update(
        {
            "model": "GAT",
            "n_tasks": 1,
            "in_node_feats": node_featurizer.feat_size(),
        }
    )

    model = GAT(
        in_feats=exp_configure["in_node_feats"],
        hidden_feats=[exp_configure["gnn_hidden_feats"]]
        * exp_configure["num_gnn_layers"],
        num_heads=[exp_configure["num_heads"]]
        * exp_configure["num_gnn_layers"],
        feat_drops=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
        attn_drops=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
        alphas=[exp_configure["alpha"]] * exp_configure["num_gnn_layers"],
        residuals=[exp_configure["residual"]] * exp_configure["num_gnn_layers"],
    )

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ## WandB setup
    wandb.init(project="test-graph-training", config=exp_configure)

    ## Train model
    model, train_loss, val_loss, test_loss = train(
        model,
        optimizer,
        loss_func,
        train_set,
        val_set,
        test_set,
        args.n_epochs,
        "cuda",
    )
    wandb.finish()

    ## Save model and losses
    if args.model_o:
        torch.save(model.state_dict(), os.path.join(args.model_o, "model.th"))
        pkl.dump(
            train_loss, open(os.path.join(args.model_o, "train_err.pkl"), "wb")
        )
        pkl.dump(
            val_loss, open(os.path.join(args.model_o, "val_err.pkl"), "wb")
        )
        pkl.dump(
            test_loss, open(os.path.join(args.model_o, "test_err.pkl"), "wb")
        )

    ## Plot loss
    if args.plot_o:
        plot_loss(
            train_loss.mean(axis=1),
            val_loss.mean(axis=1),
            test_loss.mean(axis=1),
            args.plot_o,
        )

    print("Train loss:", train_loss[-1, :].mean())
    print("Val loss:", val_loss[-1, :].mean())
    print("Test loss:", test_loss[-1, :].mean(), flush=True)


if __name__ == "__main__":
    main()
