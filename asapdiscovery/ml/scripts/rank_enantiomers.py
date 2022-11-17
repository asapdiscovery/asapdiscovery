import argparse
from glob import glob
import json
import os
import pickle as pkl
import re
import torch

from asapdiscovery.ml.dataset import DockedDataset
from asapdiscovery.data.schema import (
    ExperimentalCompoundDataUpdate,
    EnantiomerPairList,
)
from asapdiscovery.ml.utils import find_most_recent
from train_dd import (
    add_one_hot_encodings,
    add_lig_labels,
    build_model_e3nn,
    build_model_schnet,
)


def load_affinities(fn):
    """
    Load binding affinities from JSON file of
    schema.ExperimentalCompoundDataUpdate and group every 2 compounds as an
    enantiomer pair. Sort each enantiomer pair by decreasing pIC50.

    Parameters
    ----------
    fn : str
        Path of the JSON file

    Returns
    -------
    list[tuple[str]]
        List of enantiomer pair ligand compound ids. Sorted by decreasing pIC50
    set[str]
        Set of unique compound ids
    """
    ## Load experimental data. Don't need to do any filtering as that's already
    ##  been taken care of by this point
    exp_compounds = ExperimentalCompoundDataUpdate(
        **json.load(open(fn, "r"))
    ).compounds
    affinity_dict = {
        c.compound_id: c.experimental_data["pIC50"] for c in exp_compounds
    }
    all_compounds = {c.compound_id for c in exp_compounds}

    ## Pair the compounds and rank each pair
    enant_pairs = []
    for i in range(0, len(exp_compounds), 2):
        p = tuple(
            sorted(
                (
                    exp_compounds[i].compound_id,
                    exp_compounds[i + 1].compound_id,
                ),
                key=affinity_dict.get,
                reverse=True,
            )
        )
        enant_pairs.append(p)

    return (enant_pairs, all_compounds)


def load_affinities_ep(fn):
    """
    Load binding affinities from JSON file of schema.EnantiomerPairList.

    Parameters
    ----------
    fn : str
        Path of the JSON file

    Returns
    -------
    list[tuple[str]]
        List of enantiomer pair ligand compound ids. Sorted by decreasing pIC50
    set[str]
        Set of unique compound ids
    """
    ## Load experimental data. Don't need to do any filtering as that's already
    ##  been taken care of by this point
    ep_list = EnantiomerPairList(**json.load(open(fn, "r"))).pairs
    enant_pairs = [
        (ep.active.compound_id, ep.inactive.compound_id) for ep in ep_list
    ]
    all_compounds = {c for ep in enant_pairs for c in ep}

    return (enant_pairs, all_compounds)


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
    parser.add_argument(
        "-model",
        required=True,
        help="Which type of model to use (e3nn or schnet).",
    )
    parser.add_argument("-model_dir", help="Directory with trained models.")
    parser.add_argument("-model_params", help="e3nn model parameters.")
    parser.add_argument("-qm9", help="QM9 directory for pretrained model.")
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
        "-ep",
        action="store_true",
        help="Input data is in EnantiomerPairList format.",
    )
    parser.add_argument(
        "-rm_atomref",
        action="store_true",
        help="Remove atomref embedding in QM9 pretrained SchNet.",
    )

    ## Output arguments
    parser.add_argument("-o", required=True, help="Output file basename.")

    return parser.parse_args()


def main():
    args = get_args()

    ## Load the experimental affinities
    if args.ep:
        enant_pairs, all_compounds = load_affinities_ep(args.exp)
    else:
        enant_pairs, all_compounds = load_affinities(args.exp)

    ## Get all docked structures
    all_fns = glob(f"{args.i}/*complex.pdb")
    ## Extract crystal structure and compound id from file name
    re_pat = r"(Mpro-P[0-9]{4}_0[AB]).*?([A-Z]{3}-[A-Z]{3}-.*?)_complex.pdb"
    compounds = [re.search(re_pat, fn).groups() for fn in all_fns]

    ## Trim docked structures and filenames to remove compounds that don't have
    ##  experimental data (strictly speaking this shouldn't be necessary, but
    ##  just in case)
    all_fns, compounds = zip(
        *[o for o in zip(all_fns, compounds) if o[1][1] in all_compounds]
    )
    compound_dict = {c[1]: c for c in compounds}

    ## Load the dataset
    ds = DockedDataset(all_fns, compounds)

    ## Build the model
    if args.model == "e3nn":
        ## Need to add one-hot encodings to the dataset
        ds = add_one_hot_encodings(ds)

        ## Add lig labels as node attributes if requested
        if args.lig:
            ds = add_lig_labels(ds)

        ## Load model parameters
        model_params = pkl.load(open(args.model_params, "rb"))
        model = build_model_e3nn(
            100, *model_params[1:], node_attr=args.lig, dg=args.dg
        )
        model_call = lambda model, d: model(d)
    elif args.model == "schnet":
        model = build_model_schnet(
            args.qm9, args.dg, remove_atomref=args.rm_atomref
        )
        if args.dg:
            model_call = lambda model, d: model(d["z"], d["pos"], d["lig"])
        else:
            model_call = lambda model, d: model(d["z"], d["pos"])
    else:
        raise ValueError(f"Unknown model type {args.model}.")

    ## Load model weights
    if os.path.isdir(args.model_dir):
        wts_fn = find_most_recent(args.model_dir)[1]
    elif os.path.isfile(args.model_dir):
        wts_fn = args.model_dir
    print(f"Loading weights from {wts_fn}", flush=True)
    try:
        model.load_state_dict(
            torch.load(wts_fn, map_location=torch.device("cpu"))
        )
    except Exception as e:
        print(f"Could not load weights at {wts_fn}.")
        print(e)
        print("Using model weights.", flush=True)

    ## Loop through pairs, evaluate on each one, rank, check
    ## remember to use torch.no_grad()
    correct_pairs = []
    incorrect_pairs = []
    for p in enant_pairs:
        data1 = ds[compound_dict[p[0]]][1]
        data2 = ds[compound_dict[p[1]]][1]
        with torch.no_grad():
            pred1 = model_call(model, data1)
            pred2 = model_call(model, data2)
        if pred1 > pred2:
            correct_pairs.append(p)
        else:
            incorrect_pairs.append(p)
    # pIC50 = -log10(IC50)
    pkl.dump([correct_pairs, incorrect_pairs], open(args.o, "wb"))
    print(f"{len(correct_pairs)}/{len(enant_pairs)} ranked correctly.")
    print(
        f"{len(incorrect_pairs)}/{len(enant_pairs)} ranked incorrectly.",
        flush=True,
    )


if __name__ == "__main__":
    main()
