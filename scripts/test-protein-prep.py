"""
Function to test implementation of ligand filtering
"""

import sys, os, argparse

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from covid_moonshot_ml.modeling import (
    align_receptor,
    prep_receptor,
    du_to_complex,
)
from covid_moonshot_ml.datasets.utils import save_openeye_pdb


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_prot",
        required=True,
        type=str,
        help="Path to pdb file of protein to prep.",
    )
    parser.add_argument(
        "-r",
        "--ref_prot",
        required=True,
        type=str,
        help="Path to reference pdb to align to.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Path to output_dir.",
    )
    parser.add_argument(
        "-l",
        "--loop_db",
        required=False,
        type=str,
        help="Path to loop database.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    base_file_name = os.path.splitext(os.path.split(args.input_prot)[1])[0]
    print(base_file_name)
    out_name = os.path.join(args.output_dir, base_file_name)
    for mobile_chain in ["A", "B"]:
        chain_name = f"{out_name}_chain{mobile_chain}"
        initial_prot = align_receptor(
            input_prot=args.input_prot,
            ref_prot=args.ref_prot,
            dimer=True,
            mobile_chain=mobile_chain,
            ref_chain="A",
        )

        aligned_fn = f"{chain_name}_aligned.pdb"
        save_openeye_pdb(initial_prot, aligned_fn)

        site_residue = "HIS:41: :A"
        design_units = prep_receptor(
            initial_prot, site_residue=site_residue, loop_db=args.loop_db
        )
        for i, du in enumerate(design_units):
            print(i, du)
            complex_mol = du_to_complex(du)
            prepped_fn = f"{chain_name}_prepped.pdb"
            save_openeye_pdb(complex_mol, prepped_fn)


if __name__ == "__main__":
    main()
