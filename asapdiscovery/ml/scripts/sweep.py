"""
Script for a Weights & Biases hyperparameter sweep.
"""
import argparse

import json
import multiprocessing as mp
import os
import pickle as pkl
import torch
import wandb
import yaml

from asapdiscovery.ml import MSELoss
from asapdiscovery.ml.scripts.train import (
    add_one_hot_encodings,
    add_lig_labels,
)
from asapdiscovery.ml.utils import (
    build_dataset,
    build_model,
    split_dataset,
    train,
)


def build_optimizer(model):
    """
    Create optimizer object based on options in WandB config. Current options
    are Adam and SGD.

    Parameters
    ----------
    model : Union[asapdiscovery.ml.models.GAT, mtenn.model.Model]
        Model to be trained by the optimizer

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object
    """

    ## Get config
    config = wandb.config

    ## Return None (use script default) if not present
    if "optimizer" not in config:
        print("No optimizer specified, using standard Adam.", flush=True)
        return None

    ## Correct model name if needed
    optim_type = config["optimizer"].lower()

    if optim_type == "adam":
        ## Defaults from torch if not present in config
        b1 = config["b1"] if "b1" in config else 0.9
        b2 = config["b2"] if "b2" in config else 0.999
        eps = config["eps"] if "eps" in config else 1e-8
        weight_decay = config["weight_decay"] if "weight_decay" in config else 0

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=(b1, b2),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optim_type == "sgd":
        ## Defaults from torch if not present in config
        momentum = config["momentum"] if "momentum" in config else 0
        weight_decay = config["weight_decay"] if "weight_decay" in config else 0
        dampening = config["dampening"] if "dampening" in config else 0

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    return optimizer


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i", required=True, help="Input directory/glob for docked PDB files."
    )
    parser.add_argument(
        "-exp", required=True, help="JSON file giving experimental results."
    )
    parser.add_argument("-model_params", help="e3nn model parameters.")
    parser.add_argument("-qm9", help="QM9 directory for pretrained model.")
    parser.add_argument(
        "-qm9_target", type=int, default=10, help="QM9 pretrained target."
    )
    parser.add_argument(
        "-cont",
        action="store_true",
        help="Whether to restore training with most recent model weights.",
    )
    parser.add_argument(
        "-achiral", action="store_true", help="Keep only achiral molecules."
    )
    parser.add_argument("-n", default="LIG", help="Ligand residue name.")
    parser.add_argument(
        "-w",
        type=int,
        default=1,
        help="Number of workers to use for dataset loading.",
    )

    ## Output arguments
    parser.add_argument("-model_o", help="Where to save model weights.")
    parser.add_argument("-cache", help="Cache directory for dataset.")

    ## Model parameters
    parser.add_argument(
        "-model",
        required=True,
        help="Which type of model to use (e3nn or schnet).",
    )
    parser.add_argument(
        "-lig",
        action="store_true",
        help="Whether to add e3nn node attributes for ligand atoms.",
    )
    parser.add_argument(
        "-rm_atomref",
        action="store_true",
        help="Remove atomref embedding in QM9 pretrained SchNet.",
    )
    parser.add_argument(
        "-n_dist",
        type=float,
        default=5.0,
        help="Cutoff distance for node neighbors.",
    )
    parser.add_argument("-irr", help="Hidden irreps for e3nn model.")
    parser.add_argument(
        "-config", help="Model config JSON file for graph 2D model."
    )

    ## Training arguments
    parser.add_argument(
        "-n_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train for (defaults to 1000).",
    )
    parser.add_argument(
        "-device",
        default="cuda",
        help="Device to use for training (defaults to GPU).",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Group poses for the same compound into one prediction.",
    )

    ## WandB arguments
    parser.add_argument("-proj", help="WandB project name.")
    parser.add_argument("-name", help="WandB run name.")
    parser.add_argument(
        "-e",
        "--extra_config",
        nargs="+",
        help=(
            "Any extra config options to log to WandB. Can provide any "
            "number of comma-separated key-value pairs "
            "(eg --extra_config key1,val1 key2,val2 key3,val3)."
        ),
    )

    ## MTENN arguments
    parser.add_argument(
        "-strat",
        default="delta",
        help="Which strategy to use for combining model predictions.",
    )
    parser.add_argument(
        "-comb",
        help=(
            "Which combination method to use for combining grouped "
            "predictions. Only used if --grouped is set, and must be provided "
            "in that case."
        ),
    )
    parser.add_argument(
        "-pred_r", help="Readout method to use for energy predictions."
    )
    parser.add_argument(
        "-comb_r",
        help=(
            "Readout method to use for combination output. Only used if "
            "--grouped is set."
        ),
    )

    return parser.parse_args()


def main():
    ## Initialize WandB
    r = wandb.init()
    print(wandb.config, flush=True)

    args = get_args()
    print(args.i, args.exp, args.model, flush=True)

    ## Load and split dataset
    ds, exp_data = build_dataset(
        in_files=args.i,
        model_type=args.model,
        exp_fn=args.exp,
        achiral=args.achiral,
        cache_fn=args.cache,
        grouped=args.grouped,
        lig_name=args.n,
        num_workers=args.w,
    )
    ds_train, ds_val, ds_test = split_dataset(
        ds,
        wandb.config["grouped"] if "grouped" in wandb.config else args.grouped,
    )

    ## Need to augment the datasets if using e3nn
    if args.model.lower() == "e3nn":
        ## Add one-hot encodings to the dataset
        ds_train = add_one_hot_encodings(ds_train)
        ds_val = add_one_hot_encodings(ds_val)
        ds_test = add_one_hot_encodings(ds_test)

        ## Add lig labels as node attributes if requested
        if wandb.config["lig"]:
            ds_train = add_lig_labels(ds_train)
            ds_val = add_lig_labels(ds_val)
            ds_test = add_lig_labels(ds_test)

    ## Build model and set model call function
    model, model_call = build_model(
        model_type=args.model,
        e3nn_params=args.model_params,
        strat=args.strat,
        grouped=args.grouped,
        comb=args.comb,
        pred_r=args.pred_r,
        comb_r=args.comb_r,
    )

    ## Set up optimizer based on WandB config
    optimizer = build_optimizer(model)

    # print("pred", model_call(model, next(iter(ds))[1]), flush=True)

    loss_func = MSELoss("step")

    ## Initialization
    start_epoch = 0
    train_loss = []
    val_loss = []
    test_loss = []

    model_dir = os.path.join(args.model_o, r.name)
    os.makedirs(model_dir, exist_ok=True)
    model, train_loss, val_loss, test_loss = train(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        target_dict=exp_data,
        n_epochs=args.n_epochs,
        device=torch.device(args.device),
        model_call=model_call,
        loss_fn=loss_func,
        save_file=model_dir,
        lr=wandb.config["lr"],
        start_epoch=start_epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        use_wandb=True,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    main()
