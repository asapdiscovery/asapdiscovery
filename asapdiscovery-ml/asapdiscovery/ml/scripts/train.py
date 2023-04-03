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
from glob import glob
import os
import pickle as pkl

import numpy as np
import torch
from asapdiscovery.data.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    check_filelist_has_elements,
    extract_compounds_from_filenames,
)
from asapdiscovery.ml import EarlyStopping, GaussianNLLLoss, MSELoss  # noqa: E402
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
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.AllChem import Compute2DCoords, GenerateDepictionMatching2DStructure
    from rdkit.Chem.Draw import MolToImage

    table = wandb.Table(
        columns=["crystal", "compound_id", "molecule", "smiles", "pIC50"]
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

    # Log dataset splits
    for name, split in zip(ds_split_labels, ds_splits):
        table = make_wandb_table(split)
        wandb.log({f"dataset_splits/{name}": table})

    return run.id


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
    parser.add_argument("-plot_o", help="Where to save training loss plot.")
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
    parser.add_argument(
        "-es",
        "--early_stopping",
        type=int,
        help="Number of training epochs to allow with no improvement in val loss.",
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

    return parser.parse_args()


def init(args, rank=False):
    """
    Initialization steps that are common to all analyses.
    """

    # Parse model config file
    if args.sweep:
        import wandb

        wandb.init()
        model_config = dict(wandb.config)
        print("Using wandb config.", flush=True)
    elif args.config:
        model_config = parse_config(args.config)
    else:
        model_config = {}
    print("Using model config:", model_config, flush=True)

    # Override args parameters with model_config parameters\
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

        # Trim compounds and all_fns to ones that were successfully parse
        idx = [(c[0] != "NA") and (c[1] != "NA") for c in compounds]
        compounds = [c for c, i in zip(compounds, idx)]
        all_fns = [fn for fn, i in zip(all_fns, idx)]
    elif args.model.lower() != "gat":
        # If we're using a structure-based model, can't continue without structure files
        raise ValueError("-i must be specified for structure-based models")
    else:
        # Using a 2d model, so no need for structure files
        all_fns = []
        compounds = []

    # Load full dataset
    ds, exp_data = build_dataset(
        model_type=args.model,
        exp_fn=args.exp,
        all_fns=all_fns,
        compounds=compounds,
        achiral=args.achiral,
        cache_fn=args.cache,
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
        train_frac=args.tr_frac,
        val_frac=args.val_frac,
        test_frac=args.te_frac,
        rand_seed=(None if args.rand_seed else args.ds_seed),
    )

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

    model, model_call = build_model(
        model_type=args.model,
        e3nn_params=e3nn_params,
        strat=args.strat,
        grouped=args.grouped,
        comb=args.comb,
        pred_r=args.pred_r,
        comb_r=args.comb_r,
        config=model_config,
    )
    print("Model", model, flush=True)

    # Set up optimizer
    optimizer = build_optimizer(model, model_config)

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
        }
    )

    # Add MTENN options
    if (args.model.lower() == "schnet") or (args.model.lower() == "e3nn"):
        exp_configure.update(
            {
                "mtenn:strategy": args.strat,
                "mtenn:combination": args.comb,
                "mtenn:pred_readout": args.pred_r,
                "mtenn:comb_readout": args.comb_r,
            }
        )

    # Update exp_configure to have model info in it
    exp_configure.update({f"model_config:{k}": v for k, v in model_config.items()})

    # Early stopping
    if args.early_stopping:
        es = EarlyStopping(args.early_stopping)
        exp_configure.update({"early_stopping": args.early_stopping})
    else:
        es = None

    return (
        exp_data,
        ds_train,
        ds_val,
        ds_test,
        model,
        model_call,
        optimizer,
        es,
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
        es,
        exp_configure,
    ) = init(args)

    # Load model weights as necessary
    if args.cont:
        start_epoch, wts_fn = find_most_recent(args.model_o)

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
        if args.wts_fn:
            wts_fn = args.wts_fn
        else:
            wts_fn = None
        start_epoch = 0
        train_loss = None
        val_loss = None
        test_loss = None

    # Load weights
    if wts_fn:
        model = load_weights(model, wts_fn)

        # Update experiment configuration
        exp_configure.update({"wts_fn": wts_fn})

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
    else:
        raise ValueError(f"Unknown loss type {args.loss}")

    print("sq", args.sq, flush=True)
    loss_str = args.loss.lower() if args.loss else "mse"
    exp_configure.update({"loss_func": loss_str, "sq": args.sq})

    # Add any extra user-supplied config options
    if args.extra_config:
        exp_configure.update(
            {a.split(",")[0]: a.split(",")[1] for a in args.extra_config}
        )

    # Start wandb
    if args.sweep:
        import wandb

        r = wandb.init()
        model_dir = os.path.join(args.model_o, r.id)

        # Log dataset splits
        for name, split in zip(["train", "val", "test"], [ds_train, ds_val, ds_test]):
            table = make_wandb_table(split)
            wandb.log({f"dataset_splits/{name}": table})
    elif args.wandb:
        import wandb

        run_id_fn = os.path.join(args.model_o, "run_id")
        if args.proj:
            project_name = args.proj
        else:
            project_name = f"train-{args.model}"

        # Get project name
        if args.proj:
            project_name = args.proj
        else:
            project_name = f"train-{args.model}"

        # Load run_id to resume run
        if args.cont:
            run_id = open(run_id_fn).read().strip()
            wandb.init(project=project_name, id=run_id, resume="must")
            wandb.config.update(
                {"continue": True},
                allow_val_change=True,
            )
        else:
            # Get project name
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

    # Make output dir if necessary
    os.makedirs(model_dir, exist_ok=True)

    # Train the model
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
        lr=args.lr,
        start_epoch=start_epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        use_wandb=(args.wandb or args.sweep),
        batch_size=args.batch_size,
        es=es,
        optimizer=optimizer,
    )

    if args.wandb or args.sweep:
        wandb.finish()

    # Save model weights
    torch.save(model.state_dict(), f"{model_dir}/final.th")

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
