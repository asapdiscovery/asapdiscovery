"""
Script to train a 2D/3D model on COVID Moonshot data. Takes structural and
experimental measurement data as inputs, and saves trained models, train, val,
and test losses for each epoch. Also plots losses over time if the appropriate
CLI arguments are passed.
Example usage:
python train.py \
    -i complex_structure_dir/ \
    -exp experimental_measurements.json \
    -model_o trained_schnet/ \
    -plot_o trained_schnet/all_loss.png \
    -model schnet \
    -lig \
    -dg \
    -n_epochs 100 \
    --wandb \
    -proj test-model-compare
"""
import argparse
from dgllife.utils import CanonicalAtomFeaturizer
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
import json
import numpy as np
import os
import pickle as pkl
import re
import sys
import torch
from torch_geometric.nn import SchNet
from torch_geometric.datasets import QM9

from asapdiscovery.ml import (
    E3NNBind,
    GAT,
    SchNetBind,
    MSELoss,
    GaussianNLLLoss,
)
from asapdiscovery.ml.utils import (
    build_dataset,
    build_model,
    build_optimizer,
    calc_e3nn_model_info,
    find_most_recent,
    load_weights,
    parse_config,
    plot_loss,
    split_dataset,
    train,
)

import mtenn.conversion_utils
import mtenn.model


def add_one_hot_encodings(ds):
    """
    Add 100-length one-hot encoding of the atomic number for each entry in ds.
    Needed to match the expected format for e3nn model.

    Parameters
    ----------
    ds : data.dataset.DockedDataset
        Dataset to add encodings to

    Returns
    -------
    data.dataset.DockedDataset
        Dataset with one-hot encodings
    """
    for _, pose in ds:
        ## Use length 100 for one-hot encoding to account for atoms up to element
        ##  number 100
        pose["x"] = torch.nn.functional.one_hot(pose["z"] - 1, 100).float()

    return ds


def add_lig_labels(ds):
    """
    Convert boolean ligand labels into 0/1 labels. Needed to be able to pass
    ligand labels as node attributes in e3nn model.

    Parameters
    ----------
    ds : data.dataset.DockedDataset
        Dataset to add encodings to

    Returns
    -------
    data.dataset.DockedDataset
        Dataset with added ligand labels
    """
    ## Change key values for ligand labels
    for _, pose in ds:
        pose["z"] = pose["lig"].reshape((-1, 1)).float()

    return ds


def build_model_2d(model_config):
    """
    Build appropriate 2D graph model.

    Parameters
    ----------
    model_config : dict
        Model config

    Returns
    -------
    asapdiscovery.ml.models.GAT
        GAT graph model
    """

    model_config.update(
        {"in_node_feats": CanonicalAtomFeaturizer().feat_size()}
    )

    model = GAT(
        in_feats=model_config["in_node_feats"],
        hidden_feats=[model_config["gnn_hidden_feats"]]
        * model_config["num_gnn_layers"],
        num_heads=[model_config["num_heads"]] * model_config["num_gnn_layers"],
        feat_drops=[model_config["dropout"]] * model_config["num_gnn_layers"],
        attn_drops=[model_config["dropout"]] * model_config["num_gnn_layers"],
        alphas=[model_config["alpha"]] * model_config["num_gnn_layers"],
        residuals=[model_config["residual"]] * model_config["num_gnn_layers"],
    )

    return model, model_config


def build_model_e3nn(
    n_atom_types,
    num_neighbors,
    num_nodes,
    model_config=None,
    node_attr=False,
    neighbor_dist=5.0,
    irreps_hidden=None,
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
    node_attr : bool, default=False
        Whether the inputs will include node attributes (ligand labels)
    neighbor_dist : float, default=5.0
        Distance cutoff for nodes to be considered neighbors

    Returns
    -------
    mtenn.conversion_utils.e3nn.E3NN
        e3nn model created from input parameters
    """

    ## Build hidden irreps
    if model_config:
        if "irreps_0o" in model_config:
            irreps_hidden = o3.Irreps(
                [
                    (model_config["irreps_0o"], "0o"),
                    (model_config["irreps_0e"], "0e"),
                    (model_config["irreps_1o"], "1o"),
                    (model_config["irreps_1e"], "1e"),
                    (model_config["irreps_2o"], "2o"),
                    (model_config["irreps_2e"], "2e"),
                    (model_config["irreps_3o"], "3o"),
                    (model_config["irreps_3e"], "3e"),
                    (model_config["irreps_4o"], "4o"),
                    (model_config["irreps_4e"], "4e"),
                ]
            )
        else:
            irreps_hidden = o3.Irreps(
                [
                    (model_config["irreps_0"], "0o"),
                    (model_config["irreps_0"], "0e"),
                    (model_config["irreps_1"], "1o"),
                    (model_config["irreps_1"], "1e"),
                    (model_config["irreps_2"], "2o"),
                    (model_config["irreps_2"], "2e"),
                    (model_config["irreps_3"], "3o"),
                    (model_config["irreps_3"], "3e"),
                    (model_config["irreps_4"], "4o"),
                    (model_config["irreps_4"], "4e"),
                ]
            )
    ## Set up default hidden irreps if none specified
    elif irreps_hidden is None:
        irreps_hidden = [
            (mul, (l, p))
            for l, mul in enumerate([10, 3, 2, 1])
            for p in [-1, 1]
        ]

    ## Handle any conflicts and set defaults if necessary. model_config will
    ##  override any other parameters
    node_attr = (
        model_config["lig"]
        if model_config and ("lig" in model_config)
        else node_attr
    )
    irreps_edge_attr = (
        model_config["irreps_edge_attr"]
        if model_config and ("irreps_edge_attr" in model_config)
        else 3
    )
    layers = (
        model_config["layers"]
        if model_config and ("layers" in model_config)
        else 3
    )
    neighbor_dist = (
        model_config["max_radius"]
        if model_config and ("max_radius" in model_config)
        else neighbor_dist
    )
    number_of_basis = (
        model_config["number_of_basis"]
        if model_config and ("number_of_basis" in model_config)
        else 10
    )
    radial_layers = (
        model_config["radial_layers"]
        if model_config and ("radial_layers" in model_config)
        else 1
    )
    radial_neurons = (
        model_config["radial_neurons"]
        if model_config and ("radial_neurons" in model_config)
        else 128
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
        "irreps_node_attr": "1x0e" if node_attr else None,
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(irreps_edge_attr),
        "layers": layers,
        "max_radius": neighbor_dist,
        "number_of_basis": number_of_basis,
        "radial_layers": radial_layers,
        "radial_neurons": radial_neurons,
        "num_neighbors": num_neighbors,
        "num_nodes": num_nodes,
        "reduce_output": True,
    }

    return mtenn.conversion_utils.E3NN(model_kwargs=model_kwargs)


def build_model_schnet(
    model_config=None,
    qm9=None,
    qm9_target=10,
    remove_atomref=False,
    neighbor_dist=5.0,
):
    """
    Build appropriate SchNet model.

    Parameters
    ----------
    model_config : dict, optional
        Model config
    qm9 : str, optional
        Path to QM9 dataset, if starting with a QM9-pretrained model
    qm9_target : int, default=10
        Which QM9 target to use. Must be in the range of [0, 11]
    remove_atomref : bool, default=False
        Whether to remove the reference atom propoerties learned from the QM9
        dataset
    neighbor_dist : float, default=5.0
        Distance cutoff for nodes to be considered neighbors

    Returns
    -------
    mtenn.conversion_utils.SchNet
        MTENN SchNet model created from input parameters
    """

    ## Load pretrained model if requested, otherwise create a new SchNet
    if qm9 is None:
        if model_config:
            ## Get param values from config if they're there, otherwise just
            ##  use default SchNet values
            model_params = [
                "hidden_channels",
                "num_filters",
                "num_interactions",
                "num_gaussians",
                "cutoff",
                "max_num_neighbors",
                "readout",
            ]
            model_params = {
                p: model_config[p] for p in model_params if p in model_config
            }
            model = SchNet(**model_params)
        else:
            model = SchNet()
        model = mtenn.conversion_utils.SchNet(model)
    else:
        qm9_dataset = QM9(qm9)

        # target=10 is free energy (eV)
        model_qm9, _ = SchNet.from_qm9_pretrained(qm9, qm9_dataset, qm9_target)

        if remove_atomref:
            atomref = None
            ## Get rid of entries in state_dict that correspond to atomref
            wts = {
                k: v
                for k, v in model_qm9.state_dict().items()
                if "atomref" not in k
            }
        else:
            atomref = model_qm9.atomref.weight.detach().clone()
            wts = model_qm9.state_dict()

        model_params = (
            model_qm9.hidden_channels,
            model_qm9.num_filters,
            model_qm9.num_interactions,
            model_qm9.num_gaussians,
            model_qm9.cutoff,
            model_qm9.max_num_neighbors,
            model_qm9.readout,
            model_qm9.dipole,
            model_qm9.mean,
            model_qm9.std,
            atomref,
        )

        model = SchNet(*model_params)
        model.load_state_dict(wts)
        model = mtenn.conversion_utils.SchNet(model)

    ## Set interatomic cutoff (default of 10) to make the graph smaller
    if (model_config is None) or ("cutoff" not in model_config):
        model.cutoff = neighbor_dist

    return model


def make_wandb_table(ds_split):
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.AllChem import (
        Compute2DCoords,
        GenerateDepictionMatching2DStructure,
    )
    from rdkit.Chem.Draw import MolToImage
    import wandb

    table = wandb.Table(
        columns=["crystal", "compound_id", "molecule", "smiles", "pIC50"]
    )
    ## Build table and add each molecule
    for compound, d in ds_split:
        if type(compound) is tuple:
            xtal_id, compound_id = compound
            tmp_d = d
        else:
            xtal_id = ""
            compound_id = compound
            tmp_d = d[0]
        try:
            smiles = tmp_d["smiles"]
            mol = MolFromSmiles(smiles)
            Compute2DCoords(mol)
            GenerateDepictionMatching2DStructure(mol, mol)
            mol = wandb.Image(MolToImage(mol, size=(300, 300)))
        except (KeyError, ValueError):
            smiles = ""
            mol = None
        try:
            pic50 = tmp_d["pic50"].item()
        except KeyError:
            pic50 = np.nan
        except AttributeError:
            pic50 = tmp_d["pic50"]
        table.add_data(xtal_id, compound_id, mol, smiles, pic50)

    return table


def wandb_init(
    project_name,
    run_name,
    exp_configure,
    ds_splits,
    ds_split_labels=["train", "val", "test"],
):
    import wandb

    run = wandb.init(project=project_name, config=exp_configure, name=run_name)

    ## Log dataset splits
    for name, split in zip(ds_split_labels, ds_splits):
        table = make_wandb_table(split)
        wandb.log({f"dataset_splits/{name}": table})

    return run.id


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
    parser.add_argument("-plot_o", help="Where to save training loss plot.")
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
    parser.add_argument("-config", help="Model config JSON/YAML file.")
    parser.add_argument(
        "-wts_fn", help="Specific model weights file to load from."
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
        "-lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (defaults to 1e-4).",
    )
    parser.add_argument(
        "-loss",
        help="Loss type. Options are [step, uncertainty, uncertainty_sq].",
    )
    parser.add_argument(
        "-sq",
        type=float,
        help="Value to fill in for uncertainty of semiquantitative data.",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1, help="Training batch size."
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Group poses for the same compound into one prediction.",
    )

    ## WandB arguments
    parser.add_argument(
        "--wandb", action="store_true", help="Enable WandB logging."
    )
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
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="This run is part of a WandB sweep.",
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


def init(args, rank=False):
    """
    Initialization steps that are common to all analyses.
    """

    ## Parse model config file
    if args.sweep:
        import wandb

        wandb.init()
        model_config = dict(wandb.config)
    elif args.config:
        model_config = parse_config(args.config)
    else:
        model_config = {}
    print("Using model config:", model_config, flush=True)

    ## Load full dataset
    ds, exp_data = build_dataset(
        in_files=args.i,
        model_type=args.model,
        exp_fn=args.exp,
        achiral=args.achiral,
        cache_fn=args.cache,
        grouped=args.grouped,
        lig_name=args.n,
        num_workers=args.w,
        rank=rank,
    )
    ds_train, ds_val, ds_test = split_dataset(
        ds,
        model_config["grouped"] if "grouped" in model_config else args.grouped,
    )

    ## Need to augment the datasets if using e3nn
    if args.model.lower() == "e3nn":
        ## Add one-hot encodings to the dataset
        ds_train = add_one_hot_encodings(ds_train)
        ds_val = add_one_hot_encodings(ds_val)
        ds_test = add_one_hot_encodings(ds_test)

        ## Add lig labels as node attributes if requested
        if args.lig:
            ds_train = add_lig_labels(ds_train)
            ds_val = add_lig_labels(ds_val)
            ds_test = add_lig_labels(ds_test)

    model, model_call = build_model(
        model_type=args.model,
        e3nn_params=args.model_params,
        strat=args.strat,
        grouped=args.grouped,
        comb=args.comb,
        pred_r=args.pred_r,
        comb_r=args.comb_r,
        config=model_config,
    )

    ## Set up optimizer
    optimizer = build_optimizer(model, model_config)
    if "lr" in model_config:
        args.lr = model_config["lr"]

    ## Update exp_configure with model parameters
    if args.model == "e3nn":
        ## Experiment configuration
        exp_configure = {
            "model": "e3nn",
            "n_atom_types": 100,
            "num_neighbors": model_params[1],
            "num_nodes": model_params[2],
            "lig": args.lig,
            "neighbor_dist": args.n_dist,
            "irreps_hidden": args.irr,
        }
    elif args.model == "schnet":
        ## Experiment configuration
        exp_configure = {
            "model": "schnet",
            "qm9": args.qm9,
            "qm9_target": args.qm9_target,
            "rm_atomref": args.rm_atomref,
            "neighbor_dist": args.n_dist,
        }
    elif args.model == "2d":
        ## Update experiment configuration
        exp_configure.update({"model": "GAT"})
    else:
        raise ValueError(f"Unknown model type {args.model}.")

    ## Common config info
    exp_configure.update(
        {
            "train_function": "utils.train",
            "run_script": "train.py",
            "continue": args.cont,
            "train_examples": len(ds_train),
            "val_examples": len(ds_val),
            "test_examples": len(ds_test),
            "batch_size": args.batch_size,
            "device": args.device,
            "grouped": args.grouped,
        }
    )

    ## Add MTENN options
    if (args.model.lower() == "schnet") or (args.model.lower() == "e3nn"):
        exp_configure.update(
            {
                "mtenn:strategy": strategy,
                "mtenn:combination": combination,
                "mtenn:pred_readout": pred_readout,
                "mtenn:comb_readout": comb_readout,
            }
        )

    ## Update exp_configure to have model info in it
    exp_configure.update(
        {f"model_config:{k}": v for k, v in model_config.items()}
    )

    return (
        exp_data,
        ds_train,
        ds_val,
        ds_test,
        model,
        model_call,
        optimizer,
        exp_configure,
    )


def main():
    args = get_args()
    print("hidden irreps:", args.irr, flush=True)
    (
        exp_data,
        ds_train,
        ds_val,
        ds_test,
        model,
        model_call,
        optimizer,
        exp_configure,
    ) = init(args)

    ## Load model weights as necessary
    if args.cont:
        start_epoch, wts_fn = find_most_recent(args.model_o)

        ## Load error dicts
        if os.path.isfile(f"{args.model_o}/train_err.pkl"):
            train_loss = pkl.load(
                open(f"{args.model_o}/train_err.pkl", "rb")
            ).tolist()
        else:
            print("Couldn't find train loss file.", flush=True)
            train_loss = None
        if os.path.isfile(f"{args.model_o}/val_err.pkl"):
            val_loss = pkl.load(
                open(f"{args.model_o}/val_err.pkl", "rb")
            ).tolist()
        else:
            print("Couldn't find val loss file.", flush=True)
            val_loss = None
        if os.path.isfile(f"{args.model_o}/test_err.pkl"):
            test_loss = pkl.load(
                open(f"{args.model_o}/test_err.pkl", "rb")
            ).tolist()
        else:
            print("Couldn't find test loss file.", flush=True)
            test_loss = None

        ## Need to add 1 to start_epoch bc the found idx is the last epoch
        ##  successfully trained, not the one we want to start at
        start_epoch += 1
    else:
        if args.wts_fn:
            wts_fn = args.wts_fn
        else:
            wts_fn = None
        start_epoch = 0
        train_loss = None
        val_loss = None
        test_loss = None

    ## Load weights
    if wts_fn:
        model = load_weights(model, wts_fn)

        ## Update experiment configuration
        exp_configure.update({"wts_fn": wts_fn})

    ## Update experiment configuration
    exp_configure.update({"start_epoch": start_epoch})

    ## Set up the loss function
    if (args.loss is None) or (args.loss.lower() == "step"):
        loss_func = MSELoss(args.loss)
        lt = "standard" if args.loss is None else args.loss.lower()
        print(f"Using {lt} MSE loss", flush=True)
    elif "uncertainty" in args.loss.lower():
        keep_sq = "sq" in args.loss.lower()
        loss_func = GaussianNLLLoss(keep_sq, args.sq)
        print(
            f"Using Gaussian NLL loss with{'out'*(not keep_sq)}",
            "semiquant values",
            flush=True,
        )

    print("sq", args.sq, flush=True)
    loss_str = args.loss.lower() if args.loss else "mse"
    exp_configure.update({"loss_func": loss_str, "sq": args.sq})

    ## Add any extra user-supplied config options
    exp_configure.update(
        {a.split(",")[0]: a.split(",")[1] for a in args.extra_config}
    )

    ## Start wandb
    if args.sweep:
        import wandb

        r = wandb.init()
        model_dir = os.path.join(args.model_o, r.id)
    elif args.wandb:
        import wandb

        run_id_fn = os.path.join(args.model_o, "run_id")
        if args.proj:
            project_name = args.proj
        else:
            project_name = f"train-{args.model}"

        ## Get project name
        if args.proj:
            project_name = args.proj
        else:
            project_name = f"train-{args.model}"

        ## Load run_id to resume run
        if args.cont:
            run_id = open(run_id_fn).read().strip()
            run = wandb.init(project=project_name, id=run_id, resume="must")
            wandb.config.update(
                {"continue": True},
                allow_val_change=True,
            )
        else:
            ## Get project name
            run_id = wandb_init(
                project_name,
                args.name,
                exp_configure,
                [ds_train, ds_val, ds_test],
            )
            if args.model_o:
                with open(os.path.join(args.model_o, "run_id"), "w") as fp:
                    fp.write(run_id)
        model_dir = os.path.join(args.model_o, run_id)
    else:
        model_dir = args.model_o

    ## Make output dir if necessary
    os.makedirs(model_dir, exist_ok=True)

    ## Train the model
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
        save_file=args.model_o,
        lr=args.lr,
        start_epoch=start_epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        use_wandb=(args.wandb or args.sweep),
        batch_size=args.batch_size,
        optimizer=optimizer,
    )

    if args.wandb:
        wandb.finish()

    ## Plot loss
    if args.plot_o is not None:
        plot_loss(
            train_loss.mean(axis=1),
            val_loss.mean(axis=1),
            test_loss.mean(axis=1),
            args.plot_o,
        )


if __name__ == "__main__":
    main()
