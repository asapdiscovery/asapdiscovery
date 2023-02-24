import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from asapdiscovery.data.fragalysis import parse_xtal  # noqa: E402
from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate  # noqa: E402
from asapdiscovery.docking.mcs import rank_structures_openeye  # noqa: E402
from asapdiscovery.docking.mcs import rank_structures_rdkit  # noqa: 402


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-exp", required=True, help="JSON of ExperimentalCompoundDataUpdate."
    )
    parser.add_argument(
        "-x_dict",
        required=True,
        help="JSON of dict mapping compound_id: CrystalCompoundData.",
    )
    parser.add_argument(
        "-x_fn",
        required=True,
        help="CSV file with crystal compound information.",
    )
    parser.add_argument(
        "-x_dir", required=True, help="Directory with crystal structures."
    )

    # Output arguments
    parser.add_argument(
        "-o",
        required=False,
        help=(
            "Output JSON file for updated dict. If not provided, will "
            "overwrite the file given in -x_dict."
        ),
    )

    # Performance arguments
    parser.add_argument("-n", default=1, type=int, help="Number of processors to use.")
    parser.add_argument(
        "-sys",
        default="rdkit",
        help="Which package to use for MCS search [rdkit, oe].",
    )
    parser.add_argument(
        "-str",
        action="store_true",
        help=(
            "Use structure-based matching instead of element-based matching " "for MCS."
        ),
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Load experimental data
    exp_compounds = ExperimentalCompoundDataUpdate(
        **json.load(open(args.exp))
    ).compounds

    # Load mapping of compound_ids to crystal structures
    xtal_dict = json.load(open(args.x_dict))

    # Load all crystal structure information
    xtal_compounds = parse_xtal(args.x_fn, args.x_dir)

    # Lists of all xtal ids/smiles
    xtal_ids = [x.dataset for x in xtal_compounds]
    xtal_smiles = [x.smiles for x in xtal_compounds]

    # Find which compounds we need to run MCSS on
    mcs_compounds = []
    for compound in exp_compounds:
        if not xtal_dict[compound.compound_id].dataset:
            mcs_compounds.append(exp_compounds)

    # Set up MCSS
    # TODO: What if we specify something different than "rdkit" or "oe"?
    #  Might cause problems with undefined `rank_fn` variable
    if args.sys.lower() == "rdkit":
        rank_fn = rank_structures_rdkit
    elif args.sys.lower() == "oe":
        rank_fn = rank_structures_openeye
    n_procs = min(args.n, mp.cpu_count(), len(mcs_compounds))

    # Prepare the arguments to pass to starmap
    mp_args = [
        (
            c.smiles,
            c.compound_id,
            xtal_smiles,
            xtal_ids,
            None,
            args.str,
        )
        for c in exp_compounds
    ]

    # Run MCSS
    with mp.Pool(n_procs) as pool:
        sort_idxs = pool.starmap(rank_fn, mp_args)

    # Update dict with MCS matches
    for c, sort_idx in zip(mcs_compounds, sort_idxs):
        xtal = xtal_compounds[sort_idx[0]]
        xtal.sdf_fn = get_sdf_fn_from_dataset(xtal.dataset, args.x_dir)  # noqa: F821
        xtal_dict[c.compound_id] = xtal

    # Dump updated dict
    if not args.o:
        out_fn = args.x_dict
    else:
        out_fn = args.o
    json.dump(xtal_dict, open(out_fn, "w"))


if __name__ == "__main__":
    main()
