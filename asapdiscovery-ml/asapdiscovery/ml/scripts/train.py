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
from glob import glob

import numpy as np
import torch
from asapdiscovery.data.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    check_filelist_has_elements,
    extract_compounds_from_filenames,
)
from asapdiscovery.ml import (  # noqa: E402
    BestEarlyStopping,
    ConvergedEarlyStopping,
    GaussianNLLLoss,
    MSELoss,
)
from asapdiscovery.ml.utils import (
    build_dataset,
    build_loss_function,
    build_model,
    build_optimizer,
    calc_e3nn_model_info,
    find_most_recent,
    load_weights,
    parse_config,
    split_dataset,
    train,
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


def make_wandb_table(ds_split):
    import wandb

    table = wandb.Table(
        columns=[
            "crystal",
            "compound_id",
            "pIC50",
            "pIC50_range",
            "pIC50_stderr",
            "date_created",
        ]
    )
    # Build table and add each molecule
    for compound, d in ds_split:
        if type(compound) is tuple:
            xtal_id, compound_id = compound
            tmp_d = d
        else:
            xtal_id = ""
            compound_id = compound
            tmp_d = d[0]
        try:
            pic50 = tmp_d["pIC50"]
        except KeyError:
            pic50 = np.nan
        try:
            pic50_range = tmp_d["pIC50_range"]
        except KeyError:
            pic50_range = np.nan
        try:
            pic50_stderr = tmp_d["pIC50_stderr"]
        except KeyError:
            pic50_stderr = np.nan
        except AttributeError:
            pic50 = tmp_d["pIC50"]
        try:
            date_created = tmp_d["date_created"]
        except KeyError:
            date_created = None
        table.add_data(
            xtal_id, compound_id, pic50, pic50_range, pic50_stderr, date_created
        )

    return table


def wandb_init(
    project_name,
    run_name,
    sweep,
    cont,
    model_o=None,
    exp_configure=None,
):
    """
    Initialize WandB, handling saving the run ID (for continuing the run later).

    Parameters
    ----------
    project_name : str
        WandB project name
    run_name : str
        WandB run name
    sweep : bool
        Whether this is a sweep run (True) or regular training run (False)
    cont : bool
        Whether this run is a continuation of a previous run
    model_o : str, optional
        Output directory, necessary if continuing a run
    exp_configure : dict
        Dict passed for WandB config

    Returns
    -------
    str
        The WandB run ID for the initialized run
    """
    import wandb

    if sweep:
        run_id = wandb.init().id
    else:
        try:
            run_id_fn = os.path.join(model_o, "run_id")
        except TypeError:
            run_id_fn = None

        if cont:
            if run_id_fn is None:
                raise ValueError("No model_o directory specified, can't continue run.")

            # Load run_id to continue from file
            # First make sure the file exists
            try:
                run_id = open(run_id_fn).read().strip()
            except FileNotFoundError:
                raise FileNotFoundError("Couldn't find run_id file to continue run.")
            # Make sure the run_id is legit
            try:
                wandb.init(project=project_name, id=run_id, resume="must")
            except wandb.errors.UsageError:
                raise wandb.errors.UsageError(
                    f"Run in run_id file ({run_id}) doesn't exist"
                )
            # Update run config to reflect it's been resumed
            wandb.config.update(
                {"continue": True},
                allow_val_change=True,
            )
        else:
            # Start new run
            run_id = wandb.init(
                project=project_name, config=exp_configure, name=run_name
            ).id

            # Save run_id in case we want to continue later
            if run_id_fn is None:
                print(
                    "No model_o directory specified, not saving run_id anywhere.",
                    flush=True,
                )
            else:
                with open(run_id_fn, "w") as fp:
                    fp.write(run_id)

    return run_id


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument("-i", help="Input directory/glob for docked PDB files.")
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
    parser.add_argument("-cache", help="Cache directory for dataset.")

    # Dataset arguments
    parser.add_argument(
        "-tr_frac",
        type=float,
        default=0.8,
        help="Fraction of dataset to use for training.",
    )
    parser.add_argument(
        "-val_frac",
        type=float,
        default=0.1,
        help="Fraction of dataset to use for validation.",
    )
    parser.add_argument(
        "-te_frac",
        type=float,
        default=0.1,
        help="Fraction of dataset to use for testing.",
    )
    parser.add_argument(
        "-ds_seed", type=int, default=42, help="Manual seed for splitting the dataset."
    )
    parser.add_argument(
        "--rand_seed",
        action="store_true",
        help="Use a random seed for splitting the dataset. Will override -ds_seed.",
    )
    parser.add_argument(
        "-x_re",
        "--xtal_regex",
        help="Regex for extracting crystal structure name from filename.",
    )
    parser.add_argument(
        "-c_re",
        "--cpd_regex",
        help="Regex for extracting compound ID from filename.",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Split molecules temporally. Overrides random splitting.",
    )

    # Model parameters
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
    parser.add_argument("-wts_fn", help="Specific model weights file to load from.")

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
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Group poses for the same compound into one prediction.",
    )

    # Early stopping argumens
    parser.add_argument(
        "-es_t",
        "--es_type",
        help="Which early stopping strategy to use. Options are [best, converged].",
    )
    parser.add_argument(
        "-es_p",
        "--es_patience",
        type=int,
        help=(
            "Number of training epochs to allow with no improvement in val loss. "
            "Used if --es_type is best."
        ),
    )
    parser.add_argument(
        "-es_n",
        "--es_n_check",
        type=int,
        help=(
            "Number of past epoch losses to keep track of when determining "
            "convergence. Used if --es_type is converged."
        ),
    )
    parser.add_argument(
        "-es_d",
        "--es_divergence",
        type=float,
        help=(
            "Max allowable difference from the mean of the losses as a fraction of the "
            "average loss. Used if --es_type is converged."
        ),
    )

    # WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
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

    # MTENN arguments
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
    parser.add_argument("-pred_r", help="Readout method to use for energy predictions.")
    parser.add_argument(
        "-comb_r",
        help=(
            "Readout method to use for combination output. Only used if "
            "--grouped is set."
        ),
    )
    parser.add_argument(
        "-comb_neg",
        type=bool,
        default=True,
        help="Value to pass for neg when creating MaxCombination.",
    )
    parser.add_argument(
        "-comb_scale",
        type=float,
        default=1000.0,
        help="Value to pass for scale when creating MaxCombination.",
    )
    parser.add_argument(
        "-sub",
        "--substrate_conc",
        type=float,
        help=(
            "Substrate concentration for use in the Cheng-Prusoff equation. "
            "Assumed to be in the same units as Km."
        ),
    )
    parser.add_argument(
        "-km",
        "--michaelis_const",
        type=float,
        help=(
            "Km value for use in the Cheng-Prusoff equation. "
            "Assumed to be in the same units as substrate concentration."
        ),
    )

    return parser.parse_args()


def init(args, rank=False):
    """
    Initialization steps that are common to all analyses.
    """

    # Start wandb
    if args.sweep or args.wandb:
        run_id = wandb_init(args.proj, args.name, args.sweep, args.cont, args.model_o)
        model_dir = os.path.join(args.model_o, run_id)
    else:
        model_dir = args.model_o

    # Make output dir if necessary
    os.makedirs(model_dir, exist_ok=True)

    # Parse model config
    if args.sweep and args.config:
        import wandb

        # Get both configs
        sweep_config = dict(wandb.config)
        model_config = parse_config(args.config)

        # Sweep config overrules CLI config
        model_config.update(sweep_config)
    elif args.config:
        model_config = parse_config(args.config)
    elif args.sweep:
        import wandb

        model_config = dict(wandb.config)
    else:
        model_config = {}
    print("Using model config:", model_config, flush=True)

    # Override args parameters with model_config parameters
    # This shouldn't strictly be necessary, as model_config should override
    #  everything, but just to be safe
    if "grouped" in model_config:
        args.grouped = model_config["grouped"]
    if "lig" in model_config:
        args.lig = model_config["lig"]
    if "strat" in model_config:
        args.strat = model_config["strat"]
    if "comb" in model_config:
        args.comb = model_config["comb"]
    if "pred_r" in model_config:
        args.pred_r = model_config["pred_r"]
    if "comb_r" in model_config:
        args.comb_r = model_config["comb_r"]
    if "comb_neg" in model_config:
        args.comb_neg = model_config["comb_neg"]
    if "comb_scale" in model_config:
        args.comb_scale = model_config["comb_scale"]
    if "substrate" in model_config:
        args.substrate_conc = model_config["substrate"]
    if "km" in model_config:
        args.michaelis_const = model_config["km"]
    if "lr" in model_config:
        args.lr = model_config["lr"]
    if "cutoff" in model_config:
        args.n_dist = model_config["cutoff"]

    # Decide which nan values to filter
    if args.loss is None:
        # Plain MSE loss, so don't need to worry about the in range or stderr values
        check_range_nan = False
        check_stderr_nan = False
    elif args.loss.lower() == "step":
        # Step MSE loss, so only need to worry about the in range value
        check_range_nan = True
        check_stderr_nan = False
    else:
        # Using the stderr information in loss calculations, so need to include both
        check_range_nan = True
        check_stderr_nan = True

    if args.i:
        # Parse compounds from args.i
        if os.path.isdir(args.i):
            all_fns = glob(f"{args.i}/*complex.pdb")
        else:
            all_fns = glob(args.i)
        check_filelist_has_elements(all_fns, "ml_dataset")

        # Parse compound filenames
        xtal_regex = args.xtal_regex if args.xtal_regex else MPRO_ID_REGEX
        compound_regex = args.cpd_regex if args.cpd_regex else MOONSHOT_CDD_ID_REGEX
        compounds = extract_compounds_from_filenames(
            all_fns, xtal_pat=xtal_regex, compound_pat=compound_regex, fail_val="NA"
        )

        # Trim compounds and all_fns to ones that were successfully parsed
        idx = [(c[0] != "NA") and (c[1] != "NA") for c in compounds]
        compounds = [c for c, i in zip(compounds, idx) if i]
        all_fns = [fn for fn, i in zip(all_fns, idx) if i]
    elif args.model.lower() != "gat":
        # If we're using a structure-based model, can't continue without structure files
        raise ValueError("-i must be specified for structure-based models")
    else:
        # Using a 2d model, so no need for structure files
        all_fns = []
        compounds = []

    # Set cache_fn for GAT model if not already given
    if (args.model.lower() == "gat") and (not args.cache):
        cache_fn = os.path.join(model_dir, "graph.bin")
    else:
        cache_fn = args.cache

    # Load full dataset
    ds, exp_data = build_dataset(
        model_type=args.model,
        exp_fn=args.exp,
        all_fns=all_fns,
        compounds=compounds,
        achiral=args.achiral,
        cache_fn=cache_fn,
        grouped=args.grouped,
        lig_name=args.n,
        num_workers=args.w,
        rank=rank,
        check_range_nan=check_range_nan,
        check_stderr_nan=check_stderr_nan,
    )
    ds_train, ds_val, ds_test = split_dataset(
        ds,
        args.grouped,
        temporal=args.temporal,
        train_frac=args.tr_frac,
        val_frac=args.val_frac,
        test_frac=args.te_frac,
        rand_seed=(None if args.rand_seed else args.ds_seed),
    )

    if args.sweep or args.wandb:
        import wandb

        # Log dataset splits
        for name, split in zip(["train", "val", "test"], [ds_train, ds_val, ds_test]):
            table = make_wandb_table(split)
            wandb.log({f"dataset_splits/{name}": table})

    # Need to augment the datasets if using e3nn
    if args.model.lower() == "e3nn":
        # Add one-hot encodings to the dataset
        ds_train = add_one_hot_encodings(ds_train)
        ds_val = add_one_hot_encodings(ds_val)
        ds_test = add_one_hot_encodings(ds_test)

        # Add lig labels as node attributes if requested
        if args.lig:
            ds_train = add_lig_labels(ds_train)
            ds_val = add_lig_labels(ds_val)
            ds_test = add_lig_labels(ds_test)

        # Load or calculate model parameters
        if args.model_params is None:
            e3nn_params = calc_e3nn_model_info(ds_train, args.n_dist)
        elif os.path.isfile(args.model_params):
            e3nn_params = pkl.load(open(args.model_params, "rb"))
        else:
            e3nn_params = calc_e3nn_model_info(ds_train, args.n_dist)
            pkl.dump(e3nn_params, open(args.model_params, "wb"))
    else:
        e3nn_params = None

    model = build_model(
        model_type=args.model,
        e3nn_params=e3nn_params,
        strat=args.strat,
        grouped=args.grouped,
        comb=args.comb,
        pred_r=args.pred_r,
        comb_r=args.comb_r,
        comb_neg=args.comb_neg,
        comb_scale=args.comb_scale,
        substrate=args.substrate_conc,
        km=args.michaelis_const,
        config=model_config,
    )
    print("Model", model, flush=True)

    # Set up optimizer
    optimizer = build_optimizer(model, model_config)
    print("Optimizer", optimizer, flush=True)

    # Update exp_configure with model parameters
    if args.model == "e3nn":
        # Experiment configuration
        exp_configure = {
            "model": "e3nn",
            "n_atom_types": 100,
            "num_neighbors": e3nn_params[1],
            "num_nodes": e3nn_params[2],
            "lig": args.lig,
            "neighbor_dist": args.n_dist,
            "irreps_hidden": model.representation.irreps_hidden,
        }
    elif args.model == "schnet":
        # Experiment configuration
        exp_configure = {
            "model": "schnet",
            "qm9": args.qm9,
            "qm9_target": args.qm9_target,
            "rm_atomref": args.rm_atomref,
            "neighbor_dist": args.n_dist,
        }
    elif args.model == "gat":
        # Update experiment configuration
        exp_configure = {"model": "GAT"}
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
            "grouped": args.grouped,
            "substrate_conc": args.substrate_conc,
            "km": args.michaelis_const,
        }
    )

    # Add MTENN options
    if (args.model.lower() == "schnet") or (args.model.lower() == "e3nn"):
        exp_configure.update(
            {
                "mtenn:strategy": args.strat,
                "mtenn:combination": args.comb,
                "mtenn:comb_neg": args.comb_neg,
                "mtenn:comb_scale": args.comb_scale,
                "mtenn:pred_readout": args.pred_r,
                "mtenn:comb_readout": args.comb_r,
            }
        )

    # Update exp_configure to have model info in it
    exp_configure.update({f"model_config:{k}": v for k, v in model_config.items()})

    # Early stopping
    if args.es_type:
        es_type = args.es_type.lower()
        if es_type == "best":
            if args.es_patience <= 0:
                raise ValueError(
                    "Option to --es_patience must be > 0 if `best` es_type is used."
                )
            es = BestEarlyStopping(args.es_patience)
            exp_configure.update(
                {
                    "early_stopping:method": "best",
                    "early_stopping:patience": args.es_patience,
                }
            )
        elif es_type == "converged":
            if args.es_n_check <= 0:
                raise ValueError(
                    "Option to --es_n_check must be > 0 if `converged` es_type is used."
                )
            if args.es_divergence <= 0:
                raise ValueError(
                    "Option to --es_divergence must be > 0 if `converged` es_type is used."
                )
            es = ConvergedEarlyStopping(args.es_n_check, args.es_divergence)
            exp_configure.update(
                {
                    "early_stopping:method": "converged",
                    "early_stopping:n_check": args.es_n_check,
                    "early_stopping:divergence": args.es_divergence,
                }
            )
        else:
            raise ValueError(f"Unknown value for --es_type: {args.es_type}.")
    else:
        es = None

    # Dataset info
    exp_configure.update(
        {
            "train_frac": args.tr_frac,
            "val_frac": args.val_frac,
            "test_frac": args.te_frac,
        }
    )

    return (
        exp_data,
        ds_train,
        ds_val,
        ds_test,
        model,
        optimizer,
        es,
        exp_configure,
        model_dir,
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
        optimizer,
        es,
        exp_configure,
        model_dir,
    ) = init(args)

    # Load model weights as necessary
    if args.cont:
        start_epoch, wts_fn = find_most_recent(args.model_o)

        # Load error dicts
        if os.path.isfile(f"{args.model_o}/loss_dict.json"):
            loss_dict = json.load(open(f"{args.model_o}/loss_dict.json"))
        else:
            print("Couldn't find loss dict file.", flush=True)
            loss_dict = None

        # Need to add 1 to start_epoch bc the found idx is the last epoch
        #  successfully trained, not the one we want to start at
        start_epoch += 1
    else:
        if args.wts_fn:
            wts_fn = args.wts_fn
        else:
            wts_fn = None
        start_epoch = 0
        loss_dict = None

    # Load weights
    if wts_fn:
        model = load_weights(model, wts_fn)

        # Update experiment configuration
        exp_configure.update({"wts_fn": wts_fn})

    # Update experiment configuration
    exp_configure.update({"start_epoch": start_epoch})

    # Set up the loss function
    loss_func = build_loss_function(args.grouped, args.loss, args.sq)

    print("sq", args.sq, flush=True)
    loss_str = args.loss.lower() if args.loss else "mse"
    exp_configure.update({"loss_func": loss_str, "sq": args.sq})

    # Add any extra user-supplied config options
    if args.extra_config:
        exp_configure.update(
            {a.split(",")[0]: a.split(",")[1] for a in args.extra_config}
        )

    # Update wandb config before starting training
    if args.sweep or args.wandb:
        import wandb

        wandb.config.update(exp_configure, allow_val_change=True)

    # Train the model
    model, loss_dict = train(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        target_dict=exp_data,
        n_epochs=args.n_epochs,
        device=torch.device(args.device),
        grouped=args.grouped,
        loss_fn=loss_func,
        save_file=model_dir,
        lr=args.lr,
        start_epoch=start_epoch,
        loss_dict=loss_dict,
        use_wandb=(args.wandb or args.sweep),
        batch_size=args.batch_size,
        es=es,
        optimizer=optimizer,
    )

    if args.wandb or args.sweep:
        import wandb

        wandb.finish()

    # Save model weights
    torch.save(model.state_dict(), f"{model_dir}/final.th")


if __name__ == "__main__":
    main()
