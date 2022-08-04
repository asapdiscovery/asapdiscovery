"""
Build library of ligands from a dataset of holo crystal structures docked to a
different dataset of apo structures.
"""
import argparse
from glob import glob
from openeye import oechem
import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/../"
)
from covid_moonshot_ml.datasets.utils import (
    load_openeye_pdb,
    save_openeye_pdb,
    save_openeye_sdf,
    split_openeye_mol,
)
from covid_moonshot_ml.modeling import du_to_complex, make_du_from_new_lig

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments (these can be changed to eg yaml files later)
    parser.add_argument(
        "-apo",
        required=True,
        help="Wildcard string that will give all apo PDB files.",
    )
    parser.add_argument(
        "-holo",
        required=True,
        help="Wildcard string that will give all holo PDB files.",
    )
    parser.add_argument(
        "-ref",
        help=(
            "PDB file for reference structure to align all apo structure to "
            "before docking."
        ),
    )

    ## Output arguments
    parser.add_argument("-o", required=True, help="Parent output directory.")
    parser.add_argument(
        "-du",
        action="store_true",
        help="Store intermediate OEDesignUnit objects.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## Get all files and parse out a name
    all_apo_fns = glob(args.apo)
    all_apo_names = [
        os.path.splitext(os.path.basename(fn))[0] for fn in all_apo_fns
    ]
    all_holo_fns = glob(args.holo)
    all_apo_names = [
        os.path.splitext(os.path.basename(fn))[0] for fn in all_apo_fns
    ]

    ## Get ligands from all holo structures
    all_ligs = [
        split_openeye_mol(load_openeye_pdb(fn))["lig"] for fn in all_holo_fns
    ]

    ## Parse reference
    if args.ref:
        ref_prot = split_openeye_mol(load_openeye_pdb(args.ref))["pro"]
    else:
        ref_prot = None

    ## Make all design unit objects (optionally saving them), and save complex
    ##  PDB file
    for name, fn in zip(all_apo_names, all_apo_fns):
        out_dir = f"{args.o}/{name}/"
        os.makedirs(out_dir)
        ## Load and parse apo protein
        apo_prot = split_openeye_mol(load_openeye_pdb(fn))["pro"]
        for lig in all_ligs:
            out_fn = f"{out_dir}/{lig}"
            ## Make design unit
            du = make_du_from_new_lig(apo_prot, lig, ref_prot, False, False)

            ## Save if desired
            if args.du:
                oechem.OEWriteDesignUnit(f"{out_fn}.oedu", du)

            ## Get protein+lig complex in molecule form and save
            complex_mol = du_to_complex(du)
            save_openeye_pdb(complex_mol, f"{out_fn}.pdb")


if __name__ == "__main__":
    main()
