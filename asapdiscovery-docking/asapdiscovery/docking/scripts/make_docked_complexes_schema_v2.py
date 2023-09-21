"""
Make complex PDB files for docked SDF files.
"""
import argparse
from pathlib import Path

from asapdiscovery.data.openeye import load_openeye_sdfs
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.target import Target
from asapdiscovery.data.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    construct_regex_function,
)


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # I/O args
    parser.add_argument(
        "-d",
        "--docked_dir",
        type=Path,
        required=True,
        help="Top-level docking results dir.",
    )
    parser.add_argument(
        "-x", "--xtal_dir", type=Path, required=True, help="Fragalysis directory."
    )
    parser.add_argument(
        "-r", "--res_name", default="docked.sdf", help="Docked results filename."
    )
    parser.add_argument(
        "-o", "--out_name", default="bound.pdb", help="Output filename."
    )

    # Selection args
    parser.add_argument(
        "--cpd_regex",
        default=MOONSHOT_CDD_ID_REGEX,
        help="Regex for capturing compound id.",
    )
    parser.add_argument(
        "--xtal_regex",
        default=MPRO_ID_REGEX,
        help="Regex for capturing crystal id.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    compound_regex = construct_regex_function(args.cpd_regex)
    xtal_regex = construct_regex_function(args.xtal_regex)

    # Memoization of Targets
    target_dict = {}
    for fn in args.docked_dir.rglob(f"*/{args.res_name}"):
        print(fn, flush=True)
        compound_id = compound_regex(fn.parts[-2])
        xtal_id = xtal_regex(fn.parts[-2])
        all_ligs = [
            Ligand.from_oemol(mol, compound_name=compound_id)
            for mol in load_openeye_sdfs(fn)
        ]
        try:
            target = target_dict[xtal_id]
        except KeyError:
            target_fn = args.xtal_dir / "aligned" / xtal_id / f"{xtal_id}_bound.pdb"
            target = Target.from_pdb(target_fn, target_name=xtal_id)
            target_dict[xtal_id] = target

        for i, ligand in enumerate(all_ligs):
            out_fn = fn.parent / f"{fn.parts[-2]}_{i}_{args.out_name}"
            Complex(target=target, ligand=ligand).to_pdb(out_fn)


if __name__ == "__main__":
    main()
