"""
Make complex PDB files for docked SDF files.
"""

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

from asapdiscovery.data.backend.openeye import load_openeye_sdfs
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.target import Target
from asapdiscovery.data.util.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    construct_regex_function,
)


def make_docked_complex(docked_fn, xtal_dir, out_name, compound_regex, xtal_regex):
    print(docked_fn, flush=True)

    # Build regex functions if not already built
    if isinstance(compound_regex, str):
        compound_regex = construct_regex_function(compound_regex)
    if isinstance(xtal_regex, str):
        xtal_regex = construct_regex_function(xtal_regex)

    try:
        compound_id = compound_regex(docked_fn.parts[-2])
    except ValueError:
        print(f"Couldn't find compound id regex match for {str(docked_fn)}", flush=True)
        return
    try:
        xtal_id = xtal_regex(docked_fn.parts[-2])
    except ValueError:
        print(f"Couldn't find xtal id regex match for {str(docked_fn)}", flush=True)
        return

    all_ligs = [
        Ligand.from_oemol(mol, compound_name=compound_id)
        for mol in load_openeye_sdfs(docked_fn)
    ]
    target_fn = xtal_dir / "aligned" / xtal_id / f"{xtal_id}_bound.pdb"
    target = Target.from_pdb(target_fn, target_name=xtal_id)

    for i, ligand in enumerate(all_ligs):
        out_fn = docked_fn.parent / f"{docked_fn.parts[-2]}_{i}_{out_name}"
        Complex(target=target, ligand=ligand, ligand_chain="L").to_pdb(out_fn)


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

    # Multiprocessing args
    parser.add_argument(
        "-w",
        "--num_workers",
        default=1,
        type=int,
        help="Number of concurrent processes to run.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    mp_fn_partial = partial(
        make_docked_complex,
        xtal_dir=args.xtal_dir,
        out_name=args.out_name,
        compound_regex=args.cpd_regex,
        xtal_regex=args.xtal_regex,
    )

    if args.num_workers <= 1:
        for fn in args.docked_dir.rglob(f"*/{args.res_name}"):
            mp_fn_partial(fn)
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            pool.map(mp_fn_partial, args.docked_dir.rglob(f"*/{args.res_name}"))


if __name__ == "__main__":
    main()
