"""
This script combines all the fragalysis models into a single 
"""
import argparse
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import mdtraj as md
import numpy as np


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to structure directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        default="save_combined_frag_structures.log",
        help="Log file",
    )

    ## Performance arguments
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )

    return parser.parse_args()


def analyze_mp(fn, out_dir):
    full_protein_selection = "not element H and (chainid 0 or chainid 2)"
    active_site_string = "not element H and (chainid 0 or chainid 2) and (residue 140 to 145 or residue 163 or residue 172 or residue 25 to 27 or residue 41 or residue 49 or residue 54 or residue 165 to 168 or residue 189 to 192)"
    output_name = fn.stem

    ## Prepare logger
    handler = logging.FileHandler(out_dir / f"{output_name}-log.txt", mode="w")
    prep_logger = logging.getLogger(output_name)
    prep_logger.setLevel(logging.INFO)
    prep_logger.addHandler(handler)
    prep_logger.info(datetime.isoformat(datetime.now()))

    ## Check if outputs exists
    def check_output():
        for fn_suffix in ["_acive_site", "_full_protein"]:
            for extension in [".pdb", ".npy"]:
                if not (out_dir / f"{output_name}{fn_suffix}{extension}").exists():
                    return False
        return True

    if check_output():
        prep_logger.info("Output already exists!")
        return True

    prep_logger.info(f"Loading {fn}")
    pdb = md.load(fn)

    active_site_idx = pdb.topology.select(active_site_string)
    active_site = pdb.atom_slice(active_site_idx)

    full_protein_idx = pdb.topology.select(full_protein_selection)
    full_protein = pdb.atom_slice(full_protein_idx)

    prep_logger.info(f"Saving idx arrays to {out_dir}")
    np.save(out_dir / f"{output_name}_active_site.npy", active_site_idx)
    np.save(out_dir / f"{output_name}_full_protein.npy", full_protein_idx)

    prep_logger.info(f"Saving pdbs to {out_dir}")
    active_site.save(out_dir / f"{output_name}_active_site.pdb")
    full_protein.save(out_dir / f"{output_name}_full_protein.pdb")

    return True


def main():
    args = get_args()

    handler = logging.FileHandler(args.log_file, mode="w")
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(handler)

    main_logger.info(f"Finding files in {args.structure_dir}")
    fns = list(Path(args.structure_dir).glob("*/*.pdb"))
    assert len(fns) > 0, "No files were found!"
    main_logger.info(f"{len(fns)} files found")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_args = [(fn, out_dir) for fn in fns]

    main_logger.info(mp_args[0])
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    main_logger.info(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(analyze_mp, mp_args)


if __name__ == "__main__":
    main()
