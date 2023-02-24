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
import json
import os
import pickle as pkl
import re
from glob import glob

import numpy as np
import torch
from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate  # noqa: E402
from asapdiscovery.data.utils import check_filelist_has_elements  # noqa: E402
from asapdiscovery.ml import MSELoss  # noqa: E402
from asapdiscovery.ml import GAT, E3NNBind, GaussianNLLLoss, SchNetBind  # noqa: E402
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset  # noqa: E402
from asapdiscovery.ml.utils import calc_e3nn_model_info  # noqa: E402 E501
from asapdiscovery.ml.utils import find_most_recent  # noqa: 402
from asapdiscovery.ml.utils import plot_loss  # noqa: E402
from asapdiscovery.ml.utils import split_molecules, train  # noqa: E402
from dgllife.utils import CanonicalAtomFeaturizer
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet


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
        # Use length 100 for one-hot encoding to account for atoms up to element
        #  number 100
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
    # Change key values for ligand labels
    for _, pose in ds:
        pose["z"] = pose["lig"].reshape((-1, 1)).float()

    return ds


def load_exp_data(fn, achiral=False, return_compounds=False):
    """
    Load all experimental data from JSON file of
    schema.ExperimentalCompoundDataUpdate.

    Parameters
    ----------
    fn : str
        Path to JSON file
    achiral : bool, default=False
        Whether to only take achiral molecules
    return_compounds : bool, default=False
        Whether to return the compounds in addition to the experimental data

    Returns
    -------
    dict[str->dict]
        Dictionary mapping coumpound id to experimental data
    List[ExperimentalCompoundData], optional
        List of experimental compound data objects, only returned if
        `return_compounds` is True
    """
    # Load all compounds with experimental data and filter to only achiral
    #  molecules (to start)
    exp_compounds = ExperimentalCompoundDataUpdate(**json.load(open(fn))).compounds
    exp_compounds = [c for c in exp_compounds if c.achiral == achiral]

    exp_dict = {
        c.compound_id: c.experimental_data
        for c in exp_compounds
        if (
            ("pIC50" in c.experimental_data)
            and (not np.isnan(c.experimental_data["pIC50"]))
            and ("pIC50_range" in c.experimental_data)
            and (not np.isnan(c.experimental_data["pIC50_range"]))
            and ("pIC50_stderr" in c.experimental_data)
            and (not np.isnan(c.experimental_data["pIC50_stderr"]))
        )
    }

    if return_compounds:
        # Filter compounds
        exp_compounds = [c for c in exp_compounds if c.compound_id in exp_dict]
        return exp_dict, exp_compounds
    else:
        return exp_dict


def build_model_2d(config_fn):
    """
    Build appropriate 2D graph model.

    Parameters
    ----------
    config_fn : str
        Config JSON file

    Returns
    -------
    asapdiscovery.ml.models.GAT
        GAT graph model
    """

    exp_configure = json.load(open(config_fn))
    exp_configure.update({"in_node_feats": CanonicalAtomFeaturizer().feat_size()})

    model = GAT(
        in_feats=exp_configure["in_node_feats"],
        hidden_feats=[exp_configure["gnn_hidden_feats"]]
        * exp_configure["num_gnn_layers"],
        num_heads=[exp_configure["num_heads"]] * exp_configure["num_gnn_layers"],
        feat_drops=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
        attn_drops=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
        alphas=[exp_configure["alpha"]] * exp_configure["num_gnn_layers"],
        residuals=[exp_configure["residual"]] * exp_configure["num_gnn_layers"],
    )

    return model, exp_configure


def build_model_e3nn(
    n_atom_types,
    num_neighbors,
    num_nodes,
    node_attr=False,
    dg=False,
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
    dg : bool, default=False
        Whether to use E3NNBind model (True) or regular e3nn network (False)
    neighbor_dist : float, default=5.0
        Distance cutoff for nodes to be considered neighbors

    Returns
    -------
    e3nn.nn.models.gate_points_2101.Network
        e3nn/E3NNBind model created from input parameters
    """

    # Set up default hidden irreps if none specified
    if irreps_hidden is None:
        irreps_hidden = [
            (mul, (l, p))
            for l, mul in enumerate([10, 3, 2, 1])  # noqa: E741
            for p in [-1, 1]
        ]

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
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),
        "layers": 3,
        "max_radius": neighbor_dist,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": num_neighbors,
        "num_nodes": num_nodes,
        "reduce_output": True,
    }

    if dg:
        model = E3NNBind(**model_kwargs)
    else:
        model = Network(**model_kwargs)
    return model


def build_model_schnet(
    qm9=None, dg=False, qm9_target=10, remove_atomref=False, neighbor_dist=5.0
):
    """
    Build appropriate SchNet model.

    Parameters
    ----------
    qm9 : str, optional
        Path to QM9 dataset, if starting with a QM9-pretrained model
    dg : bool, default=False
        Whether to use SchNetBind model (True) or regular SchNet model (False)
    qm9_target : int, default=10
        Which QM9 target to use. Must be in the range of [0, 11]
    remove_atomref : bool, default=False
        Whether to remove the reference atom propoerties learned from the QM9
        dataset
    neighbor_dist : float, default=5.0
        Distance cutoff for nodes to be considered neighbors

    Returns
    -------
    torch_geometric.nn.SchNet
        SchNet/SchNetBind model created from input parameters
    """

    # Load pretrained model if requested, otherwise create a new SchNet
    if qm9 is None:
        if dg:
            model = SchNetBind()
        else:
            model = SchNet()
    else:
        qm9_dataset = QM9(qm9)

        # target=10 is free energy (eV)
        model_qm9, _ = SchNet.from_qm9_pretrained(qm9, qm9_dataset, qm9_target)

        if remove_atomref:
            atomref = None
            # Get rid of entries in state_dict that correspond to atomref
            wts = {
                k: v for k, v in model_qm9.state_dict().items() if "atomref" not in k
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

        if dg:
            model = SchNetBind(*model_params)
        else:
            model = SchNet(*model_params)
        model.load_state_dict(wts)

    # Set interatomic cutoff (default of 10) to make the graph smaller
    model.cutoff = neighbor_dist

    return model


def make_wandb_table(ds_split):
    import wandb
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.AllChem import Compute2DCoords, GenerateDepictionMatching2DStructure
    from rdkit.Chem.Draw import MolToImage

    table = wandb.Table(
        columns=["crystal", "compound_id", "molecule", "smiles", "pIC50"]
    )
    # Build table and add each molecule
    for (xtal_id, compound_id), d in ds_split:
        try:
            smiles = d["smiles"]
            mol = MolFromSmiles(smiles)
            Compute2DCoords(mol)
            GenerateDepictionMatching2DStructure(mol, mol)
            mol = wandb.Image(MolToImage(mol, size=(300, 300)))
        except (KeyError, ValueError):
            smiles = ""
            mol = None
        try:
            pic50 = d["pic50"].item()
        except KeyError:
            pic50 = np.nan
        except AttributeError:
            pic50 = d["pic50"]
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

    wandb.init(project=project_name, config=exp_configure, name=run_name)

    # Log dataset splits
    for name, split in zip(ds_split_labels, ds_splits):
        table = make_wandb_table(split)
        wandb.log({f"dataset_splits/{name}": table})


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
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

    # Output arguments
    parser.add_argument("-model_o", help="Where to save model weights.")
    parser.add_argument("-plot_o", help="Where to save training loss plot.")
    parser.add_argument("-cache", help="Cache directory for dataset.")

    # Model parameters
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
    parser.add_argument(
        "-n_dist",
        type=float,
        default=5.0,
        help="Cutoff distance for node neighbors.",
    )
    parser.add_argument("-irr", help="Hidden irreps for e3nn model.")
    parser.add_argument("-config", help="Model config JSON file for graph 2D model.")

    # Training arguments
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

    # WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("-proj", help="WandB project name.")
    parser.add_argument("-name", help="WandB run name.")

    return parser.parse_args()


def init(args, rank=False):
    """
    Initialization steps that are common to all analyses.
    """

    # Get all docked structures
    if os.path.isdir(args.i):
        all_fns = glob(f"{args.i}/*complex.pdb")
    else:
        all_fns = glob(args.i)

    check_filelist_has_elements(all_fns, tag="docked PDB files")

    # Extract crystal structure and compound id from file name
    xtal_pat = r"Mpro-.*?_[0-9][A-Z]"
    compound_pat = r"[A-Z]{3}-[A-Z]{3}-[0-9a-z]+-[0-9]+"

    xtal_matches = [re.search(xtal_pat, fn) for fn in all_fns]
    compound_matches = [re.search(compound_pat, fn) for fn in all_fns]
    idx = [bool(m1 and m2) for m1, m2 in zip(xtal_matches, compound_matches)]
    compounds = [
        (xtal_m.group(), compound_m.group())
        for xtal_m, compound_m, both_m in zip(xtal_matches, compound_matches, idx)
        if both_m
    ]
    num_found = len(compounds)
    # Dictionary mapping from compound_id to Mpro dataset(s)
    compound_id_dict = {}
    for xtal_structure, compound_id in compounds:
        try:
            compound_id_dict[compound_id].append(xtal_structure)
        except KeyError:
            compound_id_dict[compound_id] = [xtal_structure]

    if rank:
        exp_data = None
    elif args.model == "2d":
        # Load the experimental compounds
        exp_data, exp_compounds = load_exp_data(
            args.exp, achiral=args.achiral, return_compounds=True
        )

        # Get compounds that have both structure and experimental data (this
        #  step isn't actually necessary for performance, but allows a more
        #  fair comparison between 2D and 3D models)
        xtal_compound_ids = {c[1] for c in compounds}
        # Filter exp_compounds to make sure we have structures for them
        exp_compounds = [c for c in exp_compounds if c.compound_id in xtal_compound_ids]

        # Make cache directory as necessary
        if args.cache is None:
            cache_dir = os.path.join(args.model_o, ".cache")
        else:
            cache_dir = args.cache
        os.makedirs(cache_dir, exist_ok=True)

        # Build the dataset
        ds = GraphDataset(
            exp_compounds,
            node_featurizer=CanonicalAtomFeaturizer(),
            cache_file=os.path.join(cache_dir, "graph.bin"),
        )

        print(next(iter(ds)), flush=True)

        # Rename exp_compounds so the number kept is consistent
        compounds = exp_compounds
    elif args.cache and os.path.isfile(args.cache):
        # Load from cache
        ds = pkl.load(open(args.cache, "rb"))
        print("Loaded from cache", flush=True)

        # Still need to load the experimental affinities
        exp_data, exp_compounds = load_exp_data(
            args.exp, achiral=args.achiral, return_compounds=True
        )
    else:
        # TODO: pick up here, need to modify DockedDataset to deal with
        #  all the values in exp_data
        # Load the experimental affinities
        exp_data, exp_compounds = load_exp_data(
            args.exp, achiral=args.achiral, return_compounds=True
        )

        # Make dict to access smiles data
        smiles_dict = {}
        for c in exp_compounds:
            if c.compound_id not in compound_id_dict:
                continue
            for xtal_structure in compound_id_dict[c.compound_id]:
                smiles_dict[(xtal_structure, c.compound_id)] = c.smiles

        # Make dict to access experimental compound data
        exp_data_dict = {}
        for compound_id, d in exp_data.items():
            if compound_id not in compound_id_dict:
                continue
            for xtal_structure in compound_id_dict[compound_id]:
                exp_data_dict[(xtal_structure, compound_id)] = d

        # Trim docked structures and filenames to remove compounds that don't have
        #  experimental data
        all_fns, compounds = zip(
            *[o for o in zip(all_fns, compounds) if o[1][1] in exp_data]
        )

        # Build extra info dict
        extra_dict = {
            compound: {
                "smiles": smiles,
                "pIC50": exp_data_dict[compound]["pIC50"],
                "pIC50_range": exp_data_dict[compound]["pIC50_range"],
                "pIC50_stderr": exp_data_dict[compound]["pIC50_stderr"],
            }
            for compound, smiles in smiles_dict.items()
        }

        # Load the dataset
        ds = DockedDataset(
            all_fns,
            compounds,
            lig_resn=args.n,
            extra_dict=extra_dict,
            num_workers=args.w,
        )

        if args.cache:
            # Cache dataset
            pkl.dump(ds, open(args.cache, "wb"))

    num_kept = len(compounds)
    print(f"Kept {num_kept} out of {num_found} found structures", flush=True)

    # Split dataset into train/val/test (80/10/10 split)
    # use fixed seed for reproducibility
    ds_train, ds_val, ds_test = split_molecules(
        ds, [0.8, 0.1, 0.1], torch.Generator().manual_seed(42)
    )

    train_compound_ids = {c[1] for c, _ in ds_train}
    val_compound_ids = {c[1] for c, _ in ds_val}
    test_compound_ids = {c[1] for c, _ in ds_test}
    print(
        f"{len(ds_train)} training samples",
        f"({len(train_compound_ids)}) molecules,",
        f"{len(ds_val)} validation samples",
        f"({len(val_compound_ids)}) molecules,",
        f"{len(ds_test)} test samples",
        f"({len(test_compound_ids)}) molecules",
        flush=True,
    )

    # Build the model
    if args.model == "e3nn":
        # Need to add one-hot encodings to the dataset
        ds_train = add_one_hot_encodings(ds_train)
        ds_val = add_one_hot_encodings(ds_val)
        ds_test = add_one_hot_encodings(ds_test)

        # Load or calculate model parameters
        if args.model_params is None:
            model_params = calc_e3nn_model_info(ds_train, args.n_dist)
        elif os.path.isfile(args.model_params):
            model_params = pkl.load(open(args.model_params, "rb"))
        else:
            model_params = calc_e3nn_model_info(ds_train, args.n_dist)
            pkl.dump(model_params, open(args.model_params, "wb"))
        model = build_model_e3nn(
            100,
            *model_params[1:],
            node_attr=args.lig,
            dg=args.dg,
            neighbor_dist=args.n_dist,
            irreps_hidden=args.irr,
        )
        model_call = lambda model, d: model(d)  # noqa: E731

        # Add lig labels as node attributes if requested
        if args.lig:
            ds_train = add_lig_labels(ds_train)
            ds_val = add_lig_labels(ds_val)
            ds_test = add_lig_labels(ds_test)

        for k, v in ds_train[0][1].items():
            try:
                print(k, v.shape, flush=True)
            except AttributeError:
                print(k, v, flush=True)

        # Experiment configuration
        exp_configure = {
            "model": "e3nn",
            "n_atom_types": 100,
            "num_neighbors": model_params[1],
            "num_nodes": model_params[2],
            "lig": args.lig,
            "dg": args.dg,
            "neighbor_dist": args.n_dist,
            "irreps_hidden": args.irr,
        }
    elif args.model == "schnet":
        model = build_model_schnet(
            args.qm9,
            args.dg,
            args.qm9_target,
            args.rm_atomref,
            neighbor_dist=args.n_dist,
        )
        if args.dg:
            model_call = lambda model, d: model(d["z"], d["pos"], d["lig"])  # noqa: 731
        else:
            model_call = lambda model, d: model(d["z"], d["pos"])  # noqa: E731

        # Experiment configuration
        exp_configure = {
            "model": "schnet",
            "dg": args.dg,
            "qm9": args.qm9,
            "qm9_target": args.qm9_target,
            "rm_atomref": args.rm_atomref,
            "neighbor_dist": args.n_dist,
        }
    elif args.model == "2d":
        model, exp_configure = build_model_2d(args.config)
        model_call = lambda model, d: torch.reshape(  # noqa: E731
            model(d["g"], d["g"].ndata["h"]), (-1, 1)
        )

        # Update experiment configuration
        exp_configure.update({"model": "GAT"})
    else:
        raise ValueError(f"Unknown model type {args.model}.")

    # Common config info
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
        }
    )
    return (
        exp_data,
        ds_train,
        ds_val,
        ds_test,
        model,
        model_call,
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
        exp_configure,
    ) = init(args)

    # Load model weights as necessary
    if args.cont:
        start_epoch, wts_fn = find_most_recent(args.model_o)
        model.load_state_dict(torch.load(wts_fn))

        # Update experiment configuration
        exp_configure.update({"wts_fn": wts_fn})

        # Load error dicts
        if os.path.isfile(f"{args.model_o}/train_err.pkl"):
            train_loss = pkl.load(open(f"{args.model_o}/train_err.pkl", "rb")).tolist()
        else:
            print("Couldn't find train loss file.", flush=True)
            train_loss = None
        if os.path.isfile(f"{args.model_o}/val_err.pkl"):
            val_loss = pkl.load(open(f"{args.model_o}/val_err.pkl", "rb")).tolist()
        else:
            print("Couldn't find val loss file.", flush=True)
            val_loss = None
        if os.path.isfile(f"{args.model_o}/test_err.pkl"):
            test_loss = pkl.load(open(f"{args.model_o}/test_err.pkl", "rb")).tolist()
        else:
            print("Couldn't find test loss file.", flush=True)
            test_loss = None

        # Need to add 1 to start_epoch bc the found idx is the last epoch
        #  successfully trained, not the one we want to start at
        start_epoch += 1
    else:
        start_epoch = 0
        train_loss = None
        val_loss = None
        test_loss = None

    # Update experiment configuration
    exp_configure.update({"start_epoch": start_epoch})

    # Set up the loss function
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

    # Start wandb
    if args.wandb:
        import wandb

        # Get project name
        if args.proj:
            project_name = args.proj
        else:
            project_name = f"train-{args.model}"
        wandb_init(project_name, args.name, exp_configure, [ds_train, ds_val, ds_test])

    # Train the model
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
        args.model_o,
        args.lr,
        start_epoch,
        train_loss,
        val_loss,
        test_loss,
        args.wandb,
        args.batch_size,
    )

    if args.wandb:
        wandb.finish()

    # Plot loss
    if args.plot_o is not None:
        plot_loss(
            train_loss.mean(axis=1),
            val_loss.mean(axis=1),
            test_loss.mean(axis=1),
            args.plot_o,
        )


if __name__ == "__main__":
    main()
