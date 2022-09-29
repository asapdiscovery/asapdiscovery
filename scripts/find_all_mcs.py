import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import pickle as pkl
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from covid_moonshot_ml.docking.docking import parse_xtal
from covid_moonshot_ml.docking.mcs import (
    rank_structures_openeye,
    rank_structures_rdkit,
)
from covid_moonshot_ml.schema import (
    ExperimentalCompoundDataUpdate,
    CrystalCompoundData,
    EnantiomerPairList,
)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-exp", required=True, help="JSON file giving experimental results."
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

    return parser.parse_args()


def main():
    args = get_args()

    ## Load all compounds with experimental data and filter to only achiral
    ##  molecules (to start)
    if args.ep:
        exp_compounds = [
            c
            for ep in EnantiomerPairList(**json.load(open(args.exp, "r"))).pairs
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

    ## Find relevant crystal structures
    xtal_compounds = parse_xtal(args.x, args.x_dir)

    ## See if I can add a title to the MCS plots for the xtal id
    compound_ids = [c.compound_id for c in exp_compounds]
    xtal_ids = [x.dataset for x in xtal_compounds]
    xtal_smiles = [x.smiles for x in xtal_compounds]

    # TODO: What if we specify something different than "rdkit" or "oe"?
    #  Might cause problems with undefined `rank_fn` variable
    if args.sys.lower() == "rdkit":
        ## Convert SMILES to RDKit mol objects for MCS
        ## Need to canonicalize SMILES first because RDKit MCS seems to have
        ##  trouble with non-canon SMILES
        rank_fn = rank_structures_rdkit
    elif args.sys.lower() == "oe":
        rank_fn = rank_structures_openeye

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
            f"{args.o}/{c.compound_id}",
            10,
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
