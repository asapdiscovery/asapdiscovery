"""
Script to simply build a dataset, pickle it, and exit. CLI args for this script are
the same as the relevant args in train.py.
"""
import argparse
import os
import pickle as pkl
from glob import glob

from asapdiscovery.data.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    check_filelist_has_elements,
    extract_compounds_from_filenames,
)
from asapdiscovery.ml.utils import build_dataset


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument("-i", help="Input directory/glob for docked PDB files.")
    parser.add_argument(
        "-exp", required=True, help="JSON file giving experimental results."
    )

    # Dataset arguments
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
        "-achiral", action="store_true", help="Keep only achiral molecules."
    )
    parser.add_argument("-n", default="LIG", help="Ligand residue name.")
    parser.add_argument(
        "-w",
        type=int,
        default=1,
        help="Number of workers to use for dataset loading.",
    )
    parser.add_argument(
        "-model",
        required=True,
        help="Which type of model to use (e3nn or schnet).",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Group poses for the same compound into one prediction.",
    )
    parser.add_argument(
        "--check_range_nan",
        action="store_true",
        help="Check that the pIC50_range value is not NaN.",
    )
    parser.add_argument(
        "--check_stderr_nan",
        action="store_true",
        help="Check that the pIC50_stderr value is not NaN.",
    )

    # Output arguments
    parser.add_argument("-o", help="Output file.")

    return parser.parse_args()


def main():
    args = get_args()

    # Check that file doesn't already exist
    if os.path.isfile(args.o):
        print("File already exists, exiting.", flush=True)
        return

    if args.i:
        # Parse compounds from args.i
        if os.path.isdir(args.i):
            all_fns = glob(f"{args.i}/*complex.pdb")
        else:
            all_fns = glob(args.i)
        check_filelist_has_elements(all_fns, "build_dataset")

        # Parse compound filenames
        xtal_regex = args.xtal_regex if args.xtal_regex else MPRO_ID_REGEX
        compound_regex = args.cpd_regex if args.cpd_regex else MOONSHOT_CDD_ID_REGEX
        compounds = extract_compounds_from_filenames(
            all_fns, xtal_pat=xtal_regex, compound_pat=compound_regex, fail_val="NA"
        )

        # Trim compounds and all_fns to ones that were successfully parse
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

    # Load full dataset
    ds, _ = build_dataset(
        model_type=args.model,
        exp_fn=args.exp,
        all_fns=all_fns,
        compounds=compounds,
        achiral=args.achiral,
        cache_fn=args.o,
        grouped=args.grouped,
        lig_name=args.n,
        num_workers=args.w,
        check_range_nan=args.check_range_nan,
        check_stderr_nan=args.check_stderr_nan,
    )

    # GAT model creates a bin file, need to dump the pickle file
    if args.model.lower() == "gat":
        pkl.dump(ds, open(args.o, "wb"))


if __name__ == "__main__":
    main()
