import argparse
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from glob import glob
import json
import os
import pickle as pkl
import re
import sys
import torch
from torch_geometric.nn import SchNet
from torch_geometric.datasets import QM9

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from covid_moonshot_ml.data.dataset import DockedDataset
from covid_moonshot_ml.nn import E3NNBind, SchNetBind
from covid_moonshot_ml.schema import ExperimentalCompoundDataUpdate
from covid_moonshot_ml.utils import (
    calc_e3nn_model_info,
    find_most_recent,
    train,
    plot_loss,
)


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


def load_affinities(fn, achiral=True):
    """
    Load binding affinities from JSON file of
    schema.ExperimentalCompoundDataUpdate.

    Parameters
    ----------
    fn : str
        Path to JSON file
    achiral : bool, default=True
        Whether to only take achiral molecules

    Returns
    -------
    dict[str->float]
        Dictionary mapping coumpound id to experimental pIC50 value
    """
    ## Load all compounds with experimental data and filter to only achiral
    ##  molecules (to start)
    exp_compounds = ExperimentalCompoundDataUpdate(
        **json.load(open(fn, "r"))
    ).compounds
    exp_compounds = [
        c for c in exp_compounds if ((not achiral) or (c.achiral and achiral))
    ]

    affinity_dict = {
        c.compound_id: c.experimental_data["pIC50"]
        for c in exp_compounds
        if "pIC50" in c.experimental_data
    }

    return affinity_dict


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

    ## Set up default hidden irreps if none specified
    if irreps_hidden is None:
        irreps_hidden = [
            (mul, (l, p))
            for l, mul in enumerate([10, 3, 2, 1])
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

    ## Load pretrained model if requested, otherwise create a new SchNet
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

        if dg:
            model = SchNetBind(*model_params)
        else:
            model = SchNet(*model_params)
        model.load_state_dict(wts)

    ## Set interatomic cutoff (default of 10) to make the graph smaller
    model.cutoff = neighbor_dist

    return model


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

    ## Output arguments
    parser.add_argument("-model_o", help="Where to save model weights.")
    parser.add_argument("-plot_o", help="Where to save training loss plot.")

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
    parser.add_argument(
        "-n_dist",
        type=float,
        default=5.0,
        help="Cutoff distance for node neighbors.",
    )
    parser.add_argument("-irr", help="Hidden irreps for e3nn model.")

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

    return parser.parse_args()


def init(args, rank=False):
    """
    Initialization steps that are common to all analyses.
    """

    ## Get all docked structures
    all_fns = glob(f"{args.i}/*complex.pdb")
    ## Extract crystal structure and compound id from file name
    # re_pat = r"(Mpro-P[0-9]{4}_0[AB]).*?([A-Z]{3}-[A-Z]{3}-.*?)_complex\.pdb"
    re_pat = (
        r"(Mpro-.*?_[0-9][A-Z]).*?([A-Z]{3}-[A-Z]{3}-[0-9a-z]{8}-[0-9]+)"
        "_complex.pdb"
    )
    matches = [re.search(re_pat, fn) for fn in all_fns]
    compounds = [m.groups() for m in matches if m]
    num_found = len(compounds)

    if rank:
        exp_affinities = None
    else:
        ## Load the experimental affinities
        exp_affinities = load_affinities(args.exp)

        ## Trim docked structures and filenames to remove compounds that don't have
        ##  experimental data
        all_fns, compounds = zip(
            *[o for o in zip(all_fns, compounds) if o[1][1] in exp_affinities]
        )
    num_kept = len(compounds)
    print(f"Kept {num_kept} out of {num_found} found structures", flush=True)

    ## Load the dataset
    ds = DockedDataset(all_fns, compounds)

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

    ## Build the model
    if args.model == "e3nn":
        ## Need to add one-hot encodings to the dataset
        ds_train = add_one_hot_encodings(ds_train)
        ds_val = add_one_hot_encodings(ds_val)
        ds_test = add_one_hot_encodings(ds_test)

        ## Load or calculate model parameters
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
        model_call = lambda model, d: model(d)

        ## Add lig labels as node attributes if requested
        if args.lig:
            ds_train = add_lig_labels(ds_train)
            ds_val = add_lig_labels(ds_val)
            ds_test = add_lig_labels(ds_test)

        for k, v in ds_train[0][1].items():
            try:
                print(k, v.shape, flush=True)
            except AttributeError as e:
                print(k, v, flush=True)
    elif args.model == "schnet":
        model = build_model_schnet(
            args.qm9,
            args.dg,
            args.qm9_target,
            args.rm_atomref,
            neighbor_dist=args.n_dist,
        )
        if args.dg:
            model_call = lambda model, d: model(d["z"], d["pos"], d["lig"])
        else:
            model_call = lambda model, d: model(d["z"], d["pos"])
    else:
        raise ValueError(f"Unknown model type {args.model}.")

    return (exp_affinities, ds_train, ds_val, ds_test, model, model_call)


def main():
    args = get_args()
    print("hidden irreps:", args.irr, flush=True)
    exp_affinities, ds_train, ds_val, ds_test, model, model_call = init(args)

    ## Load model weights as necessary
    if args.cont:
        start_epoch, wts_fn = find_most_recent(args.model_o)
        model.load_state_dict(torch.load(wts_fn))

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
        start_epoch = 0
        train_loss = None
        val_loss = None
        test_loss = None

    ## Train the model
    model, train_loss, val_loss, test_loss = train(
        model,
        ds_train,
        ds_val,
        ds_test,
        exp_affinities,
        args.n_epochs,
        torch.device(args.device),
        model_call,
        args.model_o,
        args.lr,
        start_epoch,
        train_loss,
        val_loss,
        test_loss,
    )

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
