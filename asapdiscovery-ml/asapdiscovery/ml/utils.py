import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch


def build_dataset(
    model_type,
    exp_fn,
    all_fns=[],
    compounds=[],
    achiral=False,
    cache_fn=None,
    grouped=False,
    lig_name="LIG",
    num_workers=1,
    rank=False,
    structure_only=False,
    check_range_nan=True,
    check_stderr_nan=True,
):
    """
    Build a Dataset object from input structure files.

    Parameters
    ----------
    model_type : str
        Which model to create. Current options are ["gat", "schnet", "e3nn"]
    exp_fn : str
        JSON file giving experimental results
    all_fns : List[str], optional
        List of input docked PDB files
    compounds : List[Tuple[str, str]], optional
        List of (xtal_id, compound_id) that correspond 1:1 to in_files
    achiral : bool, default=False
        Only keep achiral molecules
    cache_fn : str, optional
        Dataset cache file
    grouped : bool, default=False
        Whether to group structures by ligand
    lig_name : str, default="LIG"
        Residue name for ligand atoms in PDB files
    num_workers : int, default=1
        Number of threads to use for dataset loading
    structure_only : bool, default=False
        If building a 2D dataset, whether to limit to only experimental compounds that
        also have structural data
    check_range_nan : bool, default=True
        Check that the "pIC50_range" value is not NaN
    check_stderr_nan : bool, default=True
        Check that the "pIC50_stderr" value is not NaN

    Returns
    -------
    """
    from asapdiscovery.data.utils import check_filelist_has_elements
    from asapdiscovery.ml.dataset import (
        DockedDataset,
        GraphDataset,
        GroupedDockedDataset,
    )

    # Load the experimental compounds
    exp_data, exp_compounds = load_exp_data(
        exp_fn,
        achiral=achiral,
        return_compounds=True,
        check_range_nan=check_range_nan,
        check_stderr_nan=check_stderr_nan,
    )

    # Parse structure filenames
    if (model_type.lower() != "gat") or structure_only:
        # Make sure the files passed match exist and match with compounds
        check_filelist_has_elements(all_fns, "ml_dataset")
        assert len(all_fns) == len(
            compounds
        ), "Different number of filenames and compound tuples."

        # Dictionary mapping from compound_id to Mpro dataset(s)
        compound_id_dict = {}
        for xtal_structure, compound_id in compounds:
            try:
                compound_id_dict[compound_id].append(xtal_structure)
            except KeyError:
                compound_id_dict[compound_id] = [xtal_structure]

    if rank:
        exp_data = None
    elif model_type.lower() == "gat":
        from dgllife.utils import CanonicalAtomFeaturizer

        print("load", len(exp_compounds), flush=True)

        if structure_only:
            # Get compounds that have both structure and experimental data (this
            #  step isn't actually necessary for performance, but allows a more
            #  fair comparison between 2D and 3D models)
            xtal_compound_ids = {c[1] for c in compounds}
            # Filter exp_compounds to make sure we have structures for them
            exp_compounds = [
                c for c in exp_compounds if c.compound_id in xtal_compound_ids
            ]
            print("filter", len(exp_compounds), flush=True)

        # Make cache directory as necessary
        if cache_fn is None:
            raise ValueError("Must provide cache_fn for 2d model.")
        elif os.path.isdir(cache_fn):
            os.makedirs(cache_fn, exist_ok=True)
            cache_fn = os.path.join(cache_fn, "graph.bin")

        # Build the dataset
        ds = GraphDataset(
            exp_compounds,
            node_featurizer=CanonicalAtomFeaturizer(),
            cache_file=cache_fn,
        )

        print(next(iter(ds)), flush=True)
    elif cache_fn and os.path.isfile(cache_fn):
        # Load from cache
        ds = pkl.load(open(cache_fn, "rb"))
        print("Loaded from cache", flush=True)
    else:
        # Make dicts to access smiles and date_created data
        smiles_dict = {}
        dates_dict = {}
        for c in exp_compounds:
            if c.compound_id not in compound_id_dict:
                continue
            for xtal_structure in compound_id_dict[c.compound_id]:
                smiles_dict[(xtal_structure, c.compound_id)] = c.smiles
                dates_dict[(xtal_structure, c.compound_id)] = c.date_created

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
                "date_created": dates_dict[compound],
            }
            for compound, smiles in smiles_dict.items()
        }

        # Load the dataset
        if grouped:
            ds = GroupedDockedDataset(
                all_fns,
                compounds,
                lig_resn=lig_name,
                extra_dict=extra_dict,
                num_workers=num_workers,
            )
        else:
            ds = DockedDataset(
                all_fns,
                compounds,
                lig_resn=lig_name,
                extra_dict=extra_dict,
                num_workers=num_workers,
            )

        if cache_fn:
            # Cache dataset
            pkl.dump(ds, open(cache_fn, "wb"))

    print(f"Kept {len(ds)} compounds", flush=True)

    return ds, exp_data


def build_model(
    model_type,
    e3nn_params=None,
    strat="delta",
    grouped=False,
    comb=None,
    pred_r=None,
    comb_r=None,
    config=None,
):
    """
    Dispatch function for building the correct model and setting model_call
    functions.

    Parameters
    ----------
    model_type : str
        Which model to create. Current options are ["gat", "schnet", "e3nn"]
    e3nn_params : Union[str, list], optional
        Pickle file containing model parameters for the e3nn model, or just the
        parameters themselves.
    strat : str, default="delta"
        Which strategy to use to combine protein and ligand representations.
        Current options are ["delta", "concat"]
    grouped : bool, default=False
        Whether to group structures by ligand
    comb : str, optional
        Which method to use to combine predictions for multiple poses in a
        grouped model. Current options are ["mean", "boltzmann"]
    pred_r : str, optional
        Which readout method to use for the individual pose predictions. Current
        options are ["pic50"]
    comb_r : str, optional
        Which readout method to use for the combined pose prediction. Current
        options are ["pic50"]
    config : dict, optional
        Override wandb config

    Returns
    -------
    Union[asapdiscovery.ml.models.GAT, mtenn.model.Model]
        Build model
    """

    # Correct model name if needed
    model_type = model_type.lower()

    # Get config
    if not config:
        try:
            import wandb

            config = dict(wandb.config)
            print("Using wandb config for model building.", flush=True)
        except Exception:
            config = {}

    if model_type == "gat":
        import torch

        model = build_model_2d(config)

        def model_call(model, d):
            return torch.reshape(model(d["g"], d["g"].ndata["h"]), (-1, 1))

    elif (model_type == "schnet") or (model_type == "e3nn"):
        import mtenn.conversion_utils
        import mtenn.model

        if model_type == "schnet":
            model = build_model_schnet(config)
            get_model = mtenn.conversion_utils.schnet.SchNet.get_model
        else:
            # Loadmodel parameters
            if (type(e3nn_params) is list) or (type(e3nn_params) is tuple):
                model_params = e3nn_params
            elif os.path.isfile(e3nn_params):
                model_params = pkl.load(open(e3nn_params, "rb"))
            else:
                raise ValueError(
                    "Must provide an appropriate value for e3nn_params "
                    f"(received {e3nn_params})"
                )
            model = build_model_e3nn(100, *model_params[1:], config)
            get_model = mtenn.conversion_utils.e3nn.E3NN.get_model

        # Take MTENN args from config if present, else from args
        strategy = config["strat"].lower() if "strat" in config else strat.lower()
        grouped = config["grouped"] if "grouped" in config else grouped

        # Check and parse combination
        try:
            combination = config["comb"].lower() if "comb" in config else comb.lower()
            if combination == "mean":
                combination = mtenn.model.MeanCombination()
            elif combination == "boltzmann":
                combination = mtenn.model.BoltzmannCombination()
            else:
                raise ValueError(f"Uknown value for -comb: {combination}")
        except AttributeError:
            # This will be triggered if combination is left blank
            #  (None.lower() => AttributeError)
            if grouped:
                raise ValueError(
                    "A value must be provided for -comb if --grouped is set."
                )
            combination = None

        # Check and parse pred readout
        try:
            pred_readout = (
                config["pred_r"].lower() if "pred_r" in config else pred_r.lower()
            )
            if pred_readout == "pic50":
                pred_readout = mtenn.model.PIC50Readout()
            elif pred_readout == "none":
                pred_readout = None
            else:
                raise ValueError(f"Uknown value for -pred_r: {pred_readout}")
        except AttributeError:
            pred_readout = None

        # Check and parse comb readout
        try:
            comb_readout = (
                config["comb_r"].lower() if "comb_r" in config else comb_r.lower()
            )
            if comb_readout == "pic50":
                comb_readout = mtenn.model.PIC50Readout()
            elif comb_readout == "none":
                comb_readout = None
            else:
                raise ValueError(f"Uknown value for -comb_r: {comb_readout}")
        except AttributeError:
            comb_readout = None

        # Use previously built model to construct mtenn.model.Model
        model = get_model(
            model=model,
            grouped=grouped,
            strategy=strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
            fix_device=True,
        )

        def model_call(model, d):
            return model(d)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

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

    if (type(config) is str) or (isinstance(config, Path)):
        config = parse_config(config)
    elif config is None:
        try:
            import wandb

            config = dict(wandb.config)
        except Exception:
            pass
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


def build_model_schnet(
    config=None,
    qm9=None,
    qm9_target=10,
    remove_atomref=False,
    neighbor_dist=5.0,
):
    """
    Build appropriate SchNet model.

    Parameters
    ----------
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.
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
    import mtenn.conversion_utils
    from torch_geometric.nn import SchNet

    # Parse config
    if type(config) is str:
        config = parse_config(config)
    elif config is None:
        try:
            import wandb

            config = dict(wandb.config)
        except Exception:
            pass
    elif type(config) != dict:
        raise ValueError(f"Unknown type of config: {type(config)}")

    # Load pretrained model if requested, otherwise create a new SchNet
    if qm9 is None:
        if config:
            # Get param values from config if they're there, otherwise just
            #  use default SchNet values
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
            model = SchNet(**model_params)
        else:
            model = SchNet()
        model = mtenn.conversion_utils.SchNet(model)
    else:
        from torch_geometric.datasets import QM9

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

        model = SchNet(*model_params)
        model.load_state_dict(wts)
        model = mtenn.conversion_utils.SchNet(model)

    # Set interatomic cutoff (default of 10) to make the graph smaller
    if (config is None) or ("cutoff" not in config):
        model.cutoff = neighbor_dist

    return model


def build_model_e3nn(
    n_atom_types,
    num_neighbors,
    num_nodes,
    config=None,
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
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.

    Returns
    -------
    mtenn.conversion_utils.e3nn.E3NN
        e3nn model created from input parameters
    """
    import mtenn.conversion_utils
    from e3nn.o3 import Irreps

    # Parse config
    if type(config) is str:
        config = parse_config(config)
    elif config is None:
        try:
            import wandb

            config = dict(wandb.config)
        except Exception:
            pass
    elif type(config) != dict:
        raise ValueError(f"Unknown type of config: {type(config)}")

    # Build hidden irreps
    if config:
        if "irreps_0o" in config:
            irreps_hidden = Irreps(
                [
                    (config["irreps_0o"], "0o"),
                    (config["irreps_0e"], "0e"),
                    (config["irreps_1o"], "1o"),
                    (config["irreps_1e"], "1e"),
                    (config["irreps_2o"], "2o"),
                    (config["irreps_2e"], "2e"),
                    (config["irreps_3o"], "3o"),
                    (config["irreps_3e"], "3e"),
                    (config["irreps_4o"], "4o"),
                    (config["irreps_4e"], "4e"),
                ]
            )
        else:
            irreps_hidden = Irreps(
                [
                    (config["irreps_0"], "0o"),
                    (config["irreps_0"], "0e"),
                    (config["irreps_1"], "1o"),
                    (config["irreps_1"], "1e"),
                    (config["irreps_2"], "2o"),
                    (config["irreps_2"], "2e"),
                    (config["irreps_3"], "3o"),
                    (config["irreps_3"], "3e"),
                    (config["irreps_4"], "4o"),
                    (config["irreps_4"], "4e"),
                ]
            )
        # Set up default hidden irreps if none specified
    elif irreps_hidden is None:
        irreps_hidden = [
            (mul, (l, p))  # noqa: E741
            for l, mul in enumerate([10, 3, 2, 1])  # noqa: E741
            for p in [-1, 1]  # noqa: E741
        ]

    # Handle any conflicts and set defaults if necessary. config will
    #  override any other parameters
    node_attr = config["lig"] if config and ("lig" in config) else node_attr
    irreps_edge_attr = (
        config["irreps_edge_attr"] if config and ("irreps_edge_attr" in config) else 3
    )
    layers = config["layers"] if config and ("layers" in config) else 3
    neighbor_dist = (
        config["max_radius"] if config and ("max_radius" in config) else neighbor_dist
    )
    number_of_basis = (
        config["number_of_basis"] if config and ("number_of_basis" in config) else 10
    )
    radial_layers = (
        config["radial_layers"] if config and ("radial_layers" in config) else 1
    )
    radial_neurons = (
        config["radial_neurons"] if config and ("radial_neurons" in config) else 128
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
        "irreps_edge_attr": Irreps.spherical_harmonics(irreps_edge_attr),
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


def build_optimizer(model, config=None):
    """
    Create optimizer object based on options in WandB config. Current options
    are Adam and SGD.

    Parameters
    ----------
    model : Union[asapdiscovery.ml.models.GAT, mtenn.model.Model]
        Model to be trained by the optimizer
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object
    """
    import torch

    # Parse config
    if type(config) is str:
        config = parse_config(config)
    elif config is None:
        try:
            import wandb

            config = dict(wandb.config)
            print("Using wandb config for optimizer building.", flush=True)
        except Exception:
            pass
    elif type(config) != dict:
        raise ValueError(f"Unknown type of config: {type(config)}")

    # Return None (use script default) if not present
    if "optimizer" not in config:
        print("No optimizer specified, using standard Adam.", flush=True)
        return None

    # Correct model name if needed
    optim_type = config["optimizer"].lower()

    if optim_type == "adam":
        # Defaults from torch if not present in config
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
        # Defaults from torch if not present in config
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


def calc_e3nn_model_info(ds, r):
    """
    Calculate parameters to use in creation of an e3nn model.

    Parameters
    ----------
    ds : data.dataset.DockedDataset
        Dataset of structures to use to calculate the parameters
    r : float
        Cutoff to use for neighbor atom calculations

    Returns
    -------
    int
        Number of unique atom types found in `ds`
    float
        Average number of neighbors for each node across all of `ds`
    int
        Rounded average number of nodes per structure in `ds`
    """
    from collections import Counter

    from torch_cluster import radius_graph

    num_neighbors = []
    num_nodes = []
    unique_atom_types = set()
    for _, pose in ds:
        edge_src, edge_dst = radius_graph(x=pose["pos"], r=r)
        num_neighbors.extend(Counter(edge_src.numpy()).values())
        num_nodes.append(pose["pos"].shape[0])
        unique_atom_types.update(pose["z"].tolist())

    return (
        len(unique_atom_types),
        np.mean(num_neighbors),
        round(np.mean(num_nodes)),
    )


def find_all_models(model_base):
    """
    Helper script to find all existing models in the given directory/file base
    name. If the given path is a directory, assume the files are in the format
    {epoch_index}.th. If the given path contains {}, assume that is a
    placeholder for the epoch index. If the given path is an existing file,
    just return that file.
    TODO : Rework this function to make the output more consistent

    Parameters
    ----------
    model_base : str
        Where to look for trained models

    Returns
    -------
    list[str/int]
        Returns sorted list of found epoch indexes if a directory or filename
        with placeholder is given, otherwise returns the given weights file.
    """
    import re

    if model_base is None:
        return []
    elif os.path.isdir(model_base):
        models = [
            int(fn.split(".")[0])
            for fn in os.listdir(model_base)
            if re.match(r"[0-9]+\.th", fn)
        ]
    elif "{}" in model_base:
        re_match = re.sub(r"{}", r"([0-9]+)", os.path.basename(model_base))
        models = [
            re.match(re_match, fn) for fn in os.listdir(os.path.dirname(model_base))
        ]
        models = [int(m.group(1)) for m in models if m is not None]
    elif os.path.isfile(model_base):
        return [model_base]
    else:
        return []

    return sorted(models)


def find_most_recent(model_wts):
    """
    Helper script to find the most recent of all found model weight files and
    determine the last epoch trained.

    Parameters
    ----------
    model_wts : str
        Where to look for trained models

    Returns
    -------
    int
        Which epoch of training the weights being used are from
    str
        Path to model weights file
    """
    if model_wts is None:
        return None
    elif os.path.isdir(model_wts):
        models = find_all_models(model_wts)
    elif "{}" in model_wts:
        models = find_all_models(model_wts)
        model_wts = os.path.dirname(model_wts)
    elif os.path.isfile(model_wts):
        return (0, model_wts)
    else:
        return None

    epoch_use = models[-1]
    return (epoch_use, f"{model_wts}/{epoch_use}.th")


def load_exp_data(
    fn,
    achiral=False,
    return_compounds=False,
    check_range_nan=True,
    check_stderr_nan=True,
):
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
    check_range_nan : bool, default=True
        Check that the "pIC50_range" value is not NaN
    check_stderr_nan : bool, default=True
        Check that the "pIC50_stderr" value is not NaN

    Returns
    -------
    dict[str->dict]
        Dictionary mapping coumpound id to experimental data
    List[ExperimentalCompoundData], optional
        List of experimental compound data objects, only returned if
        `return_compounds` is True
    """
    import json

    from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate

    # Load all compounds with experimental data and filter to only achiral
    #  molecules (to start)
    exp_compounds = ExperimentalCompoundDataUpdate(**json.load(open(fn))).compounds
    exp_compounds = [c for c in exp_compounds if ((not achiral) or c.achiral)]

    exp_dict = {
        c.compound_id: c.experimental_data
        for c in exp_compounds
        if (
            ("pIC50" in c.experimental_data)
            and (not np.isnan(c.experimental_data["pIC50"]))
            and ("pIC50_range" in c.experimental_data)
            and (
                (not check_range_nan)
                or (not np.isnan(c.experimental_data["pIC50_range"]))
            )
            and ("pIC50_stderr" in c.experimental_data)
            and (
                (not check_stderr_nan)
                or (not np.isnan(c.experimental_data["pIC50_stderr"]))
            )
        )
    }

    if return_compounds:
        # Filter compounds
        exp_compounds = [c for c in exp_compounds if c.compound_id in exp_dict]
        return exp_dict, exp_compounds
    else:
        return exp_dict


def check_model_compatibility(model, to_load, check_weights=False):
    """
    Checks if a PyTorch file or state_dict is compatible with a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to check.
    to_load : Union[str, Path, Dict[str, torch.Tensor]]
        The path to the PyTorch file, or a dictionary of the state dict.
    check_weights : bool, default=False
        Whether to check the weights of the model and the PyTorch file.

    Returns
    -------
    None

    """
    # Load the PyTorch file
    if isinstance(to_load, str) or isinstance(to_load, Path):
        test_state_dict = torch.load(to_load, map_location=torch.device("cpu"))

    elif isinstance(to_load, dict):
        test_state_dict = to_load
    else:
        raise ValueError(f"Invalid type of to_load: {type(to_load)}")
    # Get the state dicts of the model and the PyTorch file
    model_state_dict = model.state_dict()

    # Check the model architecture
    if set(model_state_dict.keys()) != set(test_state_dict.keys()):
        raise ValueError("Model architecture doesn't match.")

    # Check the model weights
    if check_weights:
        for key in model_state_dict.keys():
            if model_state_dict[key].shape != test_state_dict[key].shape:
                raise ValueError(
                    f'Model weights shape of "{key}" doesn\'t match the file.'
                )

            if not torch.allclose(
                model_state_dict[key], test_state_dict[key], atol=1e-4
            ):
                raise ValueError("Model weights don't match the file.")


def load_weights(model, wts_fn, check_compatibility=False):
    """
    Load weights for an MTENN model, initializing internal layers as necessary.

    Parameters
    ----------
    model: mtenn.Model
        Model to load weights into
    wts_fn: str
        Weights file to load from
    check_compatibility: bool, default=False
        Whether to check if the weights file is compatible with the model.
        May not work if using a `ConcatStrategy` block.

    Returns
    -------
    mtenn.Model
        Model with loaded weights
    """
    import torch

    # Load weights
    try:
        wts_dict = torch.load(wts_fn)
    except RuntimeError:
        wts_dict = torch.load(wts_fn, map_location="cpu")

    # Initialize linear module in ConcatStrategy
    if "strategy.reduce_nn.weight" in wts_dict:
        model.strategy.reduce_nn = torch.nn.Linear(
            wts_dict["strategy.reduce_nn.weight"].shape[1],
            wts_dict["strategy.reduce_nn.weight"].shape[0],
        )

    loaded_params = set(wts_dict.keys())
    model_params = set(model.state_dict().keys())
    print("extra parameters:", loaded_params - model_params)
    print("missing parameters:", model_params - loaded_params)

    # Get rid of extra params
    for p in loaded_params - model_params:
        del wts_dict[p]

    # Check compatibility
    if check_compatibility:
        check_model_compatibility(model, wts_dict, check_weights=False)
    # Load model parameters
    model.load_state_dict(wts_dict)
    print(f"Loaded model weights from {wts_fn}", flush=True)

    return model


def parse_config(config_fn):
    """
    Function to load a model config JSON/YAML file with the appropriate
    function.

    Parameters
    ----------
    config_fn : str
        Filename of the config file

    Returns
    -------
    dict
        Loaded config
    """

    if type(config_fn) is str:
        fn_ext = config_fn.split(".")[-1].lower()

    elif isinstance(config_fn, Path):
        fn_ext = config_fn.suffix[1:].lower()

    else:
        raise ValueError(f"Unknown config file type: {type(config_fn)}")

    if fn_ext == "json":
        import json

        model_config = json.load(open(config_fn))
    elif fn_ext in {"yaml", "yml"}:
        import yaml

        model_config = yaml.safe_load(open(config_fn))
    else:
        raise ValueError(f"Unknown config file extension: {fn_ext}")

    return model_config


def plot_loss(train_loss, val_loss, test_loss, out_fn):
    """
    Plot loss for train, val, and test sets.

    Parameters
    ----------
    train_loss : numpy.ndarray
        Loss at each epoch for train set
    val_loss : numpy.ndarray
        Loss at each epoch for validation set
    test_loss : numpy.ndarray
        Loss at each epoch for test set
    out_fn : str
        Path to save plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(nrows=3, figsize=(12, 8), sharex=True)
    sns.lineplot(x=range(len(train_loss)), y=train_loss, ax=axes[0])
    sns.lineplot(x=range(len(val_loss)), y=val_loss, ax=axes[1])
    sns.lineplot(x=range(len(test_loss)), y=test_loss, ax=axes[2])

    for ax, loss_type in zip(axes, ("Training", "Validation", "Test")):
        ax.set_ylabel(f"MSE {loss_type} Loss")
        ax.set_xlabel("Epoch")
        ax.set_title(f"MSE {loss_type} Loss")

    fig.savefig(out_fn, dpi=200, bbox_inches="tight")


def split_dataset(
    ds,
    grouped,
    temporal=False,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    rand_seed=42,
):
    """
    Split a dataset into train, val, and test splits. A warning will be raised
    if fractions don't add to 1.

    Parameters
    ----------
    ds: torch.Dataset
        Dataset object to split
    grouped: bool
        If data objects should be grouped by compound_id
    temporal: bool, default=False
        Split data temporally instead of randomly
    train_frac: float, default=0.8
        Fraction of dataset to put in the train split
    val_frac: float, default=0.1
        Fraction of dataset to put in the val split
    test_frac: float, default=0.1
        Fraction of dataset to put in the test split
    rand_seed: int, default=42
        Seed for dataset splitting

    Returns
    -------
    torch.Dataset
        Train split
    torch.Dataset
        Val split
    torch.Dataset
        Test split
    """
    import torch

    # Check that fractions add to 1
    if sum([train_frac, val_frac, test_frac]) != 1:
        from warnings import warn

        warn(
            (
                "Split fraction add to "
                f"{sum([train_frac, val_frac, test_frac]):0.2f}, not 1"
            ),
            RuntimeWarning,
        )

    print("using random seed:", rand_seed, flush=True)
    # Create generator
    if rand_seed is None:
        g = torch.Generator()
    else:
        g = torch.Generator().manual_seed(rand_seed)

    # Split dataset into train/val/test
    if grouped:
        if temporal:
            ds_train, ds_val, ds_test = split_temporal(
                ds, [train_frac, val_frac, test_frac], grouped=True
            )
            n_train = len(ds_train)
            n_val = len(ds_val)
            n_test = len(ds_test)
        else:
            ds_train, ds_val, ds_test = torch.utils.data.random_split(
                ds, [n_train, n_val, n_test], g
            )
            n_train = int(len(ds) * train_frac)
            n_val = int(len(ds) * val_frac)
            n_test = len(ds) - n_train - n_val
        print(
            (
                f"{n_train} training molecules, {n_val} validation molecules, "
                f"{n_test} testing molecules"
            ),
            flush=True,
        )
    else:
        if temporal:
            ds_train, ds_val, ds_test = split_temporal(
                ds, [train_frac, val_frac, test_frac]
            )
        else:
            ds_train, ds_val, ds_test = split_molecules(
                ds, [train_frac, val_frac, test_frac], g
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

    return ds_train, ds_val, ds_test


def split_molecules(ds, split_fracs, generator=None):
    """
    Split a dataset while keeping different poses of the same molecule in the
    same split. Naively splits based on compound_id, so if some compounds has a
    disproportionate number of poses your splits may be imbalanced.

    Parameters
    ----------
    ds : Union[cml.data.DockedDataset, cml.data.GraphDataset]
        Molecular dataset to split
    split_fracs : List[float]
        List of fraction of compounds to put in each split
    generator : torch.Generator, optional
        Torch Generator object to use for randomness. If none is supplied, use
        torch.default_generator

    Returns
    -------
    List[torch.utils.data.Subset]
        List of Subsets of original dataset
    """
    import torch

    # TODO: make this whole process more compact
    # First get all the unique compound_ids
    compound_ids_dict = {}
    for c in ds.compounds.keys():
        try:
            compound_ids_dict[c[1]].append(c)
        except KeyError:
            compound_ids_dict[c[1]] = [c]
    all_compound_ids = np.asarray(list(compound_ids_dict.keys()))

    # Set up generator
    if generator is None:
        generator = torch.default_generator

    print("splitting with random seed:", generator.initial_seed(), flush=True)
    # Shuffle the indices
    indices = torch.randperm(len(all_compound_ids), generator=generator)

    # For each Subset, grab all molecules with the included compound_ids
    all_subsets = []
    offset = 0
    # Go up to the last split so we can add anything that got left out from
    #  float rounding
    for frac in split_fracs[:-1]:
        split_len = int(np.floor(frac * len(indices)))
        incl_indices = indices[offset : offset + split_len]
        incl_compounds = all_compound_ids[incl_indices]
        offset += split_len

        # Get subset indices
        subset_idx = []
        for compound_id in incl_compounds:
            for compound in compound_ids_dict[compound_id]:
                subset_idx.extend([i for i in ds.compounds[compound]])
        all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

    # Finish up anything leftover
    incl_indices = indices[offset:]
    incl_compounds = all_compound_ids[incl_indices]

    # Get subset indices
    subset_idx = []
    for compound_id in incl_compounds:
        for compound in compound_ids_dict[compound_id]:
            subset_idx.extend([i for i in ds.compounds[compound]])
    all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

    return all_subsets


def split_temporal(ds, split_fracs, grouped=False, reverse=False, end_splits=2):
    """
    Split molecules temporally by date created. Earlier molecules will be placed in the
    training set and later molecules will be placed in the val/test sets (unless
    `reverse` is set).

    Parameters
    ----------
    ds : Union[cml.data.DockedDataset, cml.data.GraphDataset]
        Molecular dataset to split
    split_fracs : List[float]
        List of fraction of compounds to put in each split
    grouped : bool, default=False
        Splitting a GroupedDockedDataset object
    reverse : bool, default=False
        Reverse sorting of data
    end_splits : int, default=2
        How many splits (starting from the end of `split_fracs`) should be taken from
        the end of the list (most recent data, or oldest if `reverse` is True). This is
        in order to allow for the splits to not add up to 1, but still evaluate on the
        most recent data, which can be used to assess how much temporal data is required
        to predict on the most recent. Set to 0 to take all splits contiguously

    Returns
    -------
    List[torch.utils.data.Subset]
        List of Subsets of original dataset
    """
    import torch

    # Get which splits should be from the front and which should be from the end
    front_split_fracs = split_fracs[:-end_splits]
    # Reverse the end splits so it matches when we reverse everything later
    end_split_fracs = split_fracs[-end_splits:][::-1]

    # Calculate how many molecules we want covered through each split
    # By the end of each split, we should cumulatively covered the cumulative sum of all
    #  splits up to and including that one
    n_mols_front_split = np.cumsum(front_split_fracs) * len(ds)
    # Same but for reverse
    n_mols_end_split = np.cumsum(end_split_fracs) * len(ds)

    # TODO: make this whole process more compact
    # First get all the unique created dates
    dates_dict = {}
    # If we have a grouped dataset, we want to iterate through compound_ids, which will
    #  allow us to access a group of structures. Otherwise, loop through the structures
    #  directly
    if grouped:
        iter_list = ds.compound_ids
    else:
        iter_list = ds.structures
    for i, iter_item in enumerate(iter_list):
        if grouped:
            # Take the earliest date from all structures (they should all be the same,
            #  but just in case)
            all_dates = [
                s["date_created"]
                for s in ds.structures[iter_item]
                if "date_created" in s
            ]
            if len(all_dates) == 0:
                raise ValueError("Dataset doesn't contain dates.")
            else:
                date_created = min(all_dates)
        else:
            try:
                date_created = iter_item["date_created"]
            except KeyError:
                raise ValueError("Dataset doesn't contain dates.")
        try:
            dates_dict[date_created].append(i)
        except KeyError:
            dates_dict[date_created] = [i]
    all_dates = np.asarray(list(dates_dict.keys()))

    # Sort the dates
    all_dates_sorted = sorted(all_dates, reverse=reverse)
    all_dates_sorted_rev = all_dates_sorted[::-1]

    # Number of molecules for each date
    date_counts = [len(dates_dict[c]) for c in all_dates_sorted]
    date_counts_rev = date_counts[::-1]

    # Cumulative counts of dates
    # We will keep adding dates to a split until the appropratei number of molecules
    #  have been added to that split
    cum_date_counts = np.cumsum(date_counts)
    cum_date_counts_rev = np.cumsum(date_counts_rev)
    assert (
        cum_date_counts[-1] == cum_date_counts_rev[-1] == len(ds)
    ), "Something went wrong in dataset splitting."

    # For each Subset, grab all molecules with the included compound_ids
    all_subsets = []
    # Keep track of which structure indices we've seen so we don't double count in the
    #  end splits
    seen_idx = set()
    prev_idx = 0
    # If there are no end splits, go up to the last split so we can add anything that
    #  got left out from float rounding
    if end_splits == 0:
        final_split = n_mols_front_split.pop()
    else:
        final_split = None
    for n_mols in n_mols_front_split:
        # Find the first point in the cumsum where there are at least as many molecules
        #  as required for this split. We want to include molecules from all dates up to
        #  and including this date (+1 for including the date found)
        idx = np.nonzero(cum_date_counts >= n_mols)[0][0] + 1

        # Get dates to include
        incl_dates = all_dates_sorted[prev_idx:idx]

        # Get subset indices
        subset_idx = [i for d in incl_dates for i in dates_dict[d]]
        # By construction, we don't need to worry about double counting indices in the
        #  forward direction, only when we're working backward
        seen_idx.update(subset_idx)
        all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

        # Update counter
        prev_idx = idx

    # Finish up anything leftover
    if final_split:
        incl_dates = all_dates_sorted[prev_idx:]

        # Get subset indices
        subset_idx = [i for d in incl_dates for i in dates_dict[d]]
        all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

    # Add in the end splits
    prev_idx = 0
    end_subsets = []
    for n_mols in n_mols_end_split:
        # Find the first point in the cumsum where there are at least as many molecules
        #  as required for this split. We want to include molecules from all dates up to
        #  and including this date (+1 for including the date found)
        idx = np.nonzero(cum_date_counts_rev >= n_mols)[0][0] + 1

        # Get dates to include
        incl_dates = all_dates_sorted_rev[prev_idx:idx]

        # Get subset indices
        subset_idx = [i for d in incl_dates for i in dates_dict[d] if i not in seen_idx]
        # This shouldn't be necessary at this point, but just to make sure
        seen_idx.update(subset_idx)
        # Flip subset_idx so molecules are in the right order
        end_subsets.append(torch.utils.data.Subset(ds, subset_idx[::-1]))

        # Update counter
        prev_idx = idx
    # Flip end_subsets so splits are in the right order
    all_subsets.extend(end_subsets[::-1])

    return all_subsets


def train(
    model,
    ds_train,
    ds_val,
    ds_test,
    target_dict,
    n_epochs,
    device,
    model_call=lambda model, d: model(d),
    loss_fn=None,
    save_file=None,
    lr=1e-4,
    start_epoch=0,
    train_loss=None,
    val_loss=None,
    test_loss=None,
    use_wandb=False,
    batch_size=1,
    optimizer=None,
    es=None,
):
    """
    Train a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    ds_train : data.dataset.DockedDataset
        Train dataset to train on
    ds_val : data.dataset.DockedDataset
        Validation dataset to evaluate on
    ds_test : data.dataset.DockedDataset
        Test dataset to evaluate on
    target_dict : dict[str->float]
        Dictionary mapping from experimental compound_id to measured pIC50 value
    n_epochs : int
        Number of epochs to train for
    device : torch.device
        Where to run the training
    loss_fn : cml.nn.MSELoss
        Loss function that takes pred, target, in_range, and uncertainty values
        as inputs
    model_call : function(model, dict), default=lambda model, d: model(d)
        Function for calling the model. This is present to account for
        differences in calling the SchNet and e3nn models
    save_file : str, optional
        Where to save model weights and errors at each epoch. If a directory is
        passed, the weights will be saved as {epoch_idx}.th and the
        train/val/test losses will be saved as train_err.pkl, val_err.pkl, and
        test_err.pkl. If a string is passed containing {}, it will be formatted
        with the epoch number. Otherwise, the weights will be saved as the
        passed string
    lr : float, default=1e-4
        Learning rate
    start_epoch : int, default=0
        Which epoch the training is starting on. This is used when restarting
        training to ensure the appropriate number of epochs is reached
    train_loss : list[float], default=None
        List of train losses from previous epochs. Used when restarting training
    val_loss : list[float], default=None
        List of val losses from previous epochs. Used when restarting training
    test_loss : list[float], default=None
        List of test losses from previous epochs. Used when restarting training
    use_wandb : bool, default=False
        Log results with WandB
    batch_size : int, default=1
        Number of samples to predict on before performing backprop
    optimizer : torch.optim.Optimizer, optional
        Optimizer to use for model training. If not used, defaults to Adam
        optimizer
    es : asapdiscovery.ml.EarlyStopping
        EarlyStopping object to keep track of early stopping

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
    from time import time

    import torch

    from . import MSELoss

    if use_wandb:
        import wandb

    if train_loss is None:
        train_loss = []
    if val_loss is None:
        val_loss = []
    if test_loss is None:
        test_loss = []

    # Send model to desired device if it's not there already
    model = model.to(device)

    # Save initial model weights for debugging
    if os.path.isdir(save_file):
        torch.save(model.state_dict(), os.path.join(save_file, "init.th"))

    # Set up optimizer and loss function
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr)
    print("Using optimizer", optimizer, flush=True)

    if loss_fn is None:
        loss_fn = MSELoss()
    print("Using loss function", loss_fn, flush=True)

    # Train for n epochs
    for epoch_idx in range(start_epoch, n_epochs):
        print(f"Epoch {epoch_idx}/{n_epochs}", flush=True)
        if epoch_idx % 10 == 0 and epoch_idx > 0:
            print(f"Training error: {np.mean(train_loss[-1]):0.5f}")
            print(f"Validation error: {np.mean(val_loss[-1]):0.5f}")
            print(f"Testing error: {np.mean(test_loss[-1]):0.5f}", flush=True)
        tmp_loss = []

        # Initialize batch
        batch_counter = 0
        optimizer.zero_grad()
        batch_loss = None
        start_time = time()
        for compound, pose in ds_train:
            if type(compound) is tuple:
                compound_id = compound[1]
            else:
                compound_id = compound

            # convert to float to match other types
            target = torch.tensor(
                [[target_dict[compound_id]["pIC50"]]], device=device
            ).float()
            in_range = torch.tensor(
                [[target_dict[compound_id]["pIC50_range"]]], device=device
            ).float()
            uncertainty = torch.tensor(
                [[target_dict[compound_id]["pIC50_stderr"]]], device=device
            ).float()

            # Make prediction and calculate loss
            pred = model_call(model, pose).reshape(target.shape)
            loss = loss_fn(pred, target, in_range, uncertainty)

            # Keep track of loss for each sample
            tmp_loss.append(loss.item())

            # Update batch_loss
            if batch_loss is None:
                batch_loss = loss
            else:
                batch_loss += loss
            batch_counter += 1

            # Perform backprop if we've done all the preds for this batch
            if batch_counter == batch_size:
                # Backprop
                batch_loss.backward()
                optimizer.step()

                # Reset batch tracking
                batch_counter = 0
                optimizer.zero_grad()
                batch_loss = None

        if batch_counter > 0:
            # Backprop for final incomplete batch
            batch_loss.backward()
            optimizer.step()
        end_time = time()

        train_loss.append(np.asarray(tmp_loss))
        epoch_train_loss = np.mean(tmp_loss)

        with torch.no_grad():
            tmp_loss = []
            for compound, pose in ds_val:
                if type(compound) is tuple:
                    compound_id = compound[1]
                else:
                    compound_id = compound

                # convert to float to match other types
                target = torch.tensor(
                    [[target_dict[compound_id]["pIC50"]]], device=device
                ).float()
                in_range = torch.tensor(
                    [[target_dict[compound_id]["pIC50_range"]]], device=device
                ).float()
                uncertainty = torch.tensor(
                    [[target_dict[compound_id]["pIC50_stderr"]]], device=device
                ).float()

                # Make prediction and calculate loss
                pred = model_call(model, pose).reshape(target.shape)
                loss = loss_fn(pred, target, in_range, uncertainty)
                tmp_loss.append(loss.item())
            val_loss.append(np.asarray(tmp_loss))
            epoch_val_loss = np.mean(tmp_loss)

            tmp_loss = []
            for compound, pose in ds_test:
                if type(compound) is tuple:
                    compound_id = compound[1]
                else:
                    compound_id = compound

                # convert to float to match other types
                target = torch.tensor(
                    [[target_dict[compound_id]["pIC50"]]], device=device
                ).float()
                in_range = torch.tensor(
                    [[target_dict[compound_id]["pIC50_range"]]], device=device
                ).float()
                uncertainty = torch.tensor(
                    [[target_dict[compound_id]["pIC50_stderr"]]], device=device
                ).float()

                # Make prediction and calculate loss
                pred = model_call(model, pose).reshape(target.shape)
                loss = loss_fn(pred, target, in_range, uncertainty)
                tmp_loss.append(loss.item())
            test_loss.append(np.asarray(tmp_loss))
            epoch_test_loss = np.mean(tmp_loss)

        if use_wandb:
            wandb.log(
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "test_loss": epoch_test_loss,
                    "epoch": epoch_idx,
                    "epoch_time": end_time - start_time,
                }
            )
        if save_file is None:
            continue
        elif os.path.isdir(save_file):
            torch.save(model.state_dict(), f"{save_file}/{epoch_idx}.th")
            pkl.dump(np.vstack(train_loss), open(f"{save_file}/train_err.pkl", "wb"))
            pkl.dump(np.vstack(val_loss), open(f"{save_file}/val_err.pkl", "wb"))
            pkl.dump(np.vstack(test_loss), open(f"{save_file}/test_err.pkl", "wb"))
        elif "{}" in save_file:
            torch.save(model.state_dict(), save_file.format(epoch_idx))
        else:
            torch.save(model.state_dict(), save_file)

        # Stop if loss has gone to infinity or is NaN
        if (
            np.isnan(epoch_val_loss)
            or (epoch_val_loss == np.inf)
            or (epoch_val_loss == -np.inf)
        ):
            if os.path.isdir(save_file):
                pkl.dump(
                    ds_train,
                    open(os.path.join(save_file, "ds_train.pkl"), "wb"),
                )
                pkl.dump(
                    ds_val,
                    open(os.path.join(save_file, "ds_val.pkl"), "wb"),
                )
                pkl.dump(
                    ds_test,
                    open(os.path.join(save_file, "ds_test.pkl"), "wb"),
                )
            raise ValueError("Unrecoverable loss value reached.")

        # Stop training if EarlyStopping says to
        if es and es.check(epoch_idx, epoch_val_loss, model.state_dict()):
            print(
                (
                    f"Stopping training after epoch {epoch_idx}, "
                    f"using weights from epoch {es.best_epoch}"
                ),
                flush=True,
            )
            model.load_state_dict(es.best_wts)
            break

    return (
        model,
        np.vstack(train_loss),
        np.vstack(val_loss),
        np.vstack(test_loss),
    )
