import argparse
import json
import multiprocessing as mp
import numpy as np
import pickle as pkl

from asapdiscovery.data.utils import load_exp_from_sdf
from asapdiscovery.data.fragalysis import parse_xtal
from asapdiscovery.docking.mcs import (
    rank_structures_openeye,
    rank_structures_rdkit,
)
from asapdiscovery.data.schema import (
    ExperimentalCompoundDataUpdate,
    EnantiomerPairList,
)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument("-exp", help="JSON file giving experimental results.")
    parser.add_argument(
        "-sdf", help="SDF file containing compounds to use as queries in MCS."
    )
    parser.add_argument(
        "-x", required=True, help="CSV file with crystal compound information."
    )
    parser.add_argument(
        "-x_dir", required=True, help="Directory with crystal structures."
    )

    ## Output arguments
    parser.add_argument("-o", required=True, help="Main output directory.")

    ## Performance arguments
    parser.add_argument(
        "-n", default=1, type=int, help="Number of processors to use."
    )
    parser.add_argument(
        "-sys",
        default="rdkit",
        help="Which package to use for MCS search [rdkit, oe].",
    )
    parser.add_argument(
        "-str",
        action="store_true",
        help=(
            "Use "
            "structure-based matching instead of element-based matching for MCS."
        ),
    )
    parser.add_argument(
        "-ep",
        action="store_true",
        help="Input data is in EnantiomerPairList format.",
    )
    parser.add_argument(
        "-achiral", action="store_true", help="Only keep achiral compounds."
    )
    parser.add_argument(
        "-n_draw",
        type=int,
        default=10,
        help="Number of MCS compounds to draw for each query molecule.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## Load all compounds with experimental data and filter to only achiral
    ##  molecules (to start)
    if args.exp:
        ## Give a warning if both -exp and -sdf are provided
        if args.sdf:
            print(
                "WARNING: Both -exp and -sdf provided, using -exp.", flush=True
            )
        if args.ep:
            exp_compounds = [
                c
                for ep in EnantiomerPairList(
                    **json.load(open(args.exp, "r"))
                ).pairs
                for c in (ep.active, ep.inactive)
            ]
        else:
            exp_compounds = [
                c
                for c in ExperimentalCompoundDataUpdate(
                    **json.load(open(args.exp, "r"))
                ).compounds
                if c.smiles is not None
            ]
            exp_compounds = np.asarray(
                [
                    c
                    for c in exp_compounds
                    if ((args.achiral and c.achiral) or (not args.achiral))
                ]
            )
    elif args.sdf:
        exp_compounds = load_exp_from_sdf(args.sdf)
    else:
        raise ValueError("Need to specify exactly one of -exp or -sdf.")

    ## Find relevant crystal structures
    xtal_compounds = parse_xtal(args.x, args.x_dir)

    ## See if I can add a title to the MCS plots for the xtal id
    compound_ids = [c.compound_id for c in exp_compounds]
    xtal_ids = [x.dataset for x in xtal_compounds]
    xtal_smiles = [x.smiles for x in xtal_compounds]

    if args.sys.lower() == "rdkit":
        ## Convert SMILES to RDKit mol objects for MCS
        ## Need to canonicalize SMILES first because RDKit MCS seems to have
        ##  trouble with non-canon SMILES
        rank_fn = rank_structures_rdkit
    elif args.sys.lower() == "oe":
        rank_fn = rank_structures_openeye
    else:
        raise ValueError(f"Unknwon option for -sys: {args.sys}.")

    print(f"{len(exp_compounds)} experimental compounds")
    print(f"{len(xtal_compounds)} crystal structures")
    print("Finding best docking structures", flush=True)
    ## Prepare the arguments to pass to starmap
    mp_args = [
        (
            c.smiles,
            c.compound_id,
            xtal_smiles,
            xtal_ids,
            None,
            args.str,
            None if args.n_draw == 0 else f"{args.o}/{c.compound_id}",
            args.n_draw,
        )
        for c in exp_compounds
    ]

    n_procs = min(args.n, mp.cpu_count(), len(exp_compounds))
    with mp.Pool(n_procs) as pool:
        res = pool.starmap(rank_fn, mp_args)

    pkl.dump(
        [compound_ids, xtal_ids, res],
        open(f"{args.o}/mcs_sort_index.pkl", "wb"),
    )


if __name__ == "__main__":
    main()
