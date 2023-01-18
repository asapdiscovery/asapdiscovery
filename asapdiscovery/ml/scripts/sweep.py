"""
Script for a Weights & Biases hyperparameter sweep.
"""
import argparse

import json
import multiprocessing as mp
import pickle as pkl
import torch
import wandb
import yaml

from asapdiscovery.ml import MSELoss
from asapdiscovery.ml.scripts.train import (
    add_one_hot_encodings,
    add_lig_labels,
    build_dataset,
    split_dataset,
)
from asapdiscovery.ml.utils import train


def build_model(args):
    """
    Dispatch function for building the correct model and setting model_call
    functions.

    Parameters
    ----------
    args : dict
        CLI arguments

    Returns
    -------
    Union[asapdiscovery.ml.models.GAT, mtenn.model.Model]
        Build model
    """

    ## Correct model name if needed
    model = args.model.lower()
    ## Get config
    config = wandb.config

    if model == "2d":
        model = build_model_2d()
        model_call = lambda model, d: torch.reshape(
            model(d["g"], d["g"].ndata["h"]), (-1, 1)
        )
    elif (model == "schnet") or (model == "e3nn"):
        import mtenn.conversion_utils
        import mtenn.model

        if model == "schnet":
            model = build_model_schnet()
        else:
            model_params = pkl.load(open(args.model_params, "rb"))
            model = build_model_e3nn(100, *model_params[1:])
        strategy = args.strat.lower()

        ## Check and parse combination
        try:
            combination = args.comb.lower()
            if combination == "mean":
                combination = mtenn.model.MeanCombination()
            elif combination == "boltzmann":
                combination = mtenn.model.BoltzmannCombination()
            else:
                raise ValueError(f"Uknown value for -comb: {args.comb}")
        except AttributeError:
            ## This will be triggered if combination is left blank
            ##  (None.lower() => AttributeError)
            if args.grouped:
                raise ValueError(
                    f"A value must be provided for -comb if --grouped is set."
                )
            combination = None

        ## Check and parse pred readout
        try:
            pred_readout = args.pred_r.lower()
            if pred_readout == "pic50":
                pred_readout = mtenn.model.PIC50Readout()
            else:
                raise ValueError(f"Uknown value for -pred_r: {args.pred_r}")
        except AttributeError:
            pred_readout = None

        ## Check and parse comb readout
        try:
            comb_readout = args.comb_r.lower()
            if comb_readout == "pic50":
                comb_readout = mtenn.model.PIC50Readout()
            else:
                raise ValueError(f"Uknown value for -comb_r: {args.comb_r}")
        except AttributeError:
            comb_readout = None

        ## Use previously built model to construct mtenn.model.Model
        model = mtenn.conversion_utils.schnet.SchNet.get_model(
            model=model,
            grouped=args.grouped,
            strategy=strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
            fix_device=True,
        )
        model_call = lambda model, d: model(d)

    return model, model_call


def build_model_2d(config=None):
    """
    Build appropriate 2D graph model.

    Parameters
    ----------
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.

    Returns
    -------
    asapdiscovery.ml.models.GAT
        GAT graph model
    """
    from asapdiscovery.ml import GAT
    from dgllife.utils import CanonicalAtomFeaturizer

    if type(config) is str:
        config = json.load(open(config_fn))
    elif config is None:
        config = wandb.config
    elif type(config) != dict:
        raise ValueError(f"Unknown type of config: {type(config)}")

    # config.update({"in_node_feats": CanonicalAtomFeaturizer().feat_size()})
    in_node_feats = CanonicalAtomFeaturizer().feat_size()

    model = GAT(
        in_feats=in_node_feats,
        hidden_feats=[config["gnn_hidden_feats"]] * config["num_gnn_layers"],
        num_heads=[config["num_heads"]] * config["num_gnn_layers"],
        feat_drops=[config["dropout"]] * config["num_gnn_layers"],
        attn_drops=[config["dropout"]] * config["num_gnn_layers"],
        alphas=[config["alpha"]] * config["num_gnn_layers"],
        residuals=[config["residual"]] * config["num_gnn_layers"],
    )

    return model


def build_model_schnet(config=None):
    """
    Build appropriate SchNet model.

    Parameters
    ----------
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.

    Returns
    -------
    mtenn.conversion_utils.SchNet
        MTENN SchNet model created from input parameters
    """
    import mtenn.conversion_utils
    from torch_geometric.nn import SchNet

    ## Parse config
    if type(config) is str:
        config = json.load(open(config_fn))
    elif config is None:
        config = wandb.config
    elif type(config) != dict:
        raise ValueError(f"Unknown type of config: {type(config)}")

    ## Get param values from config if they're there, otherwise just use default
    ##  SchNet values
    model_params = [
        "hidden_channels",
        "num_filters",
        "num_interactions",
        "num_gaussians",
        "cutoff",
        "max_num_neighbors",
        "readout",
    ]
    model_params = {p: config[p] for p in model_params if p in config}

    ## Build SchNet model and then MTENN model
    model = SchNet(**model_params)
    model = mtenn.conversion_utils.SchNet(model)

    return model


def build_model_e3nn(
    n_atom_types,
    num_neighbors,
    num_nodes,
    config=None,
):
    """
    Build appropriate e3nn model.

    Parameters
    ----------
    n_atom_types : int
        Number off atom types in one-hot encodings. This will define the
        dimensionality of the input into the model
    num_neighbors : int
        Approximate number of neighbor nodes that get convolved over for each
        node. Used as a normalization factor in the model
    num_nodes : int
        Approximate number of nodes per graph. Used as a normalization factor in
        the model
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.

    Returns
    -------
    mtenn.conversion_utils.e3nn.E3NN
        e3nn model created from input parameters
    """
    from e3nn.o3 import Irreps
    import mtenn.conversion_utils

    ## Parse config
    if type(config) is str:
        config = json.load(open(config_fn))
    elif config is None:
        config = wandb.config
    elif type(config) != dict:
        raise ValueError(f"Unknown type of config: {type(config)}")

    ## Build hidden irreps
    irreps_hidden = Irreps(
        [
            (config.irreps_0o, "0o"),
            (config.irreps_0e, "0e"),
            (config.irreps_1o, "1o"),
            (config.irreps_1e, "1e"),
            (config.irreps_2o, "2o"),
            (config.irreps_2e, "2e"),
            (config.irreps_3o, "3o"),
            (config.irreps_3e, "3e"),
            (config.irreps_4o, "4o"),
            (config.irreps_4e, "4e"),
        ]
    )

    # input is one-hot encoding of atom type => n_atom_types scalars
    # output is scalar valued binding energy/pIC50 value
    # hidden layers taken from e3nn tutorial (should be tuned eventually)
    # same with edge attribute irreps (and all hyperparameters)
    # need to calculate num_neighbors and num_nodes
    # reduce_output because we just want the one binding energy prediction
    #  across the whole graph
    model_kwargs = {
        "irreps_in": f"{n_atom_types}x0e",
        "irreps_hidden": irreps_hidden,
        "irreps_out": "1x0e",
        "irreps_node_attr": "1x0e" if config.lig else None,
        "irreps_edge_attr": Irreps.spherical_harmonics(config.irreps_edge_attr),
        "layers": config.layers,
        "max_radius": config.max_radius,
        "number_of_basis": config.number_of_basis,
        "radial_layers": config.radial_layers,
        "radial_neurons": config.radial_neurons,
        "num_neighbors": num_neighbors,
        "num_nodes": num_nodes,
        "reduce_output": True,
    }

    return mtenn.conversion_utils.E3NN(model_kwargs=model_kwargs)


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
    wandb.init()
    print(wandb.config, flush=True)

    args = get_args()
    print(args.i, args.exp, args.model, flush=True)

    ## Load and split dataset
    ds, exp_data = build_dataset(args)
    ds_train, ds_val, ds_test = split_dataset(ds, args.grouped)

    ## Need to augment the datasets if using e3nn
    if args.model.lower() == "e3nn":
        ## Add one-hot encodings to the dataset
        ds_train = add_one_hot_encodings(ds_train)
        ds_val = add_one_hot_encodings(ds_val)
        ds_test = add_one_hot_encodings(ds_test)

        ## Add lig labels as node attributes if requested
        if wandb.config.lig:
            ds_train = add_lig_labels(ds_train)
            ds_val = add_lig_labels(ds_val)
            ds_test = add_lig_labels(ds_test)

    ## Build model and set model call function
    model, model_call = build_model(args)

    # print("pred", model_call(model, next(iter(ds))[1]), flush=True)

    loss_func = MSELoss("step")

    ## Initialization
    start_epoch = 0
    train_loss = []
    val_loss = []
    test_loss = []

    model, train_loss, val_loss, test_loss = train(
        model,
        ds_train,
        ds_val,
        ds_test,
        exp_data,
        args.n_epochs,
        torch.device(args.device),
        model_call,
        loss_func,
        None,
        wandb.config["lr"],
        start_epoch,
        train_loss,
        val_loss,
        test_loss,
        use_wandb=True,
    )


if __name__ == "__main__":
    main()
