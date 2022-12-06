"""
The goal of this script is to retroactively calculate the RMSDs of a set of
docking results to their corresponding reference structures.

Example Usage
    python calculate_RMSD_of_docked_ligands.py
        -sdf ~/asap-datasets/retro_docking/sars_fragalysis_retrospective/20221202/combined.sdf
        -o ~/asap-datasets/retro_docking/sars_fragalysis_retrospective/20221202
        -r ~/asap-datasets/prospective/prepped_mpro_structures_fragalysis/*/prepped_complex.pdb
"""
import argparse
from glob import glob
import multiprocessing as mp
import pandas as pd
from openeye import oechem
import sys, os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
from asapdiscovery.data.openeye import (
    load_openeye_sdfs,
    load_openeye_sdf,
    load_openeye_pdb,
    split_openeye_mol,
)
from asapdiscovery.docking.analysis import calculate_rmsd_openeye


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-sdf", "--sdf_fn", required=False, help="Path to combined sdf file."
    )
    parser.add_argument(
        "-g",
        "--sdf_glob",
        required=False,
        type=str,
        help="Expression representing individual sdf file name strings.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output directory in which will be put the output csv file",
    )
    parser.add_argument(
        "-r",
        "--ref_glob",
        type=str,
        help="Expression representing the reference structures.",
    )
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    ## Either load all from one big sdf file or from a glob that represents many
    if args.sdf_fn:
        print(f"Loading molecules from {args.sdf_fn}")
        mols = load_openeye_sdfs(args.sdf_fn)
    elif args.sdf_glob:
        print(f"Loading molecules using {args.sdf_glob}")
        mols = [load_openeye_sdf(sdf_fn) for sdf_fn in glob(args.sdf_glob)]
    else:
        raise NotImplementedError("Must pass either -sdf or -g flag")

    print(f"Loaded {len(mols)} molecules")

    ## get unique compound_ids
    compound_ids = [oechem.OEGetSDData(mol, f"Compound_ID") for mol in mols]
    unique_compound_ids = list(set(compound_ids))
    print(
        f"Using {len(compound_ids)} compound ids to find reference structures"
    )

    ## TODO: Maybe something better would be to just pass a
    ## TODO: yaml file that maps compound_ids to desired reference structures

    ## is it an sdf or a pdb?
    ref_fns = glob(args.ref_glob)
    ref_type = args.ref_glob[-3:]
    if ref_type == "pdb":
        print("Loading reference PDBs")

        ## This maps each compound id to the corresponding reference
        ref_mols = {
            compound_id: split_openeye_mol(load_openeye_pdb(ref_fn))["lig"]
            for compound_id in unique_compound_ids
            for ref_fn in ref_fns
            if compound_id in ref_fn
        }
    else:
        raise NotImplementedError(f"Sorry I've only done this for PDBs")
    print(f"{len(ref_mols)} references found")
    mp_args = []
    complex_ids = []
    final_compound_ids = []

    ## Now map each input sdf file to a reference
    for query_mol in mols:
        compound_id = oechem.OEGetSDData(query_mol, f"Compound_ID")
        try:
            ref_mol = ref_mols[compound_id]
            final_compound_ids.append(compound_id)
            complex_ids.append(query_mol.GetTitle())
            mp_args.append([query_mol, ref_mol])
        except KeyError:
            print(f"Skipping missing reference structure: {compound_id}")

    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Running {len(mp_args)} docking runs over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        rmsds = pool.starmap(calculate_rmsd_openeye, mp_args)
    df = pd.DataFrame(
        {
            "Compound_ID": final_compound_ids,
            "RMSD": rmsds,
            "Complex_ID": complex_ids,
        },
    )
    output_path = os.path.join(args.output_dir, "rmsds.csv")
    print(f"Writing results to {output_path}")
    df.to_csv(output_path)


if __name__ == "__main__":
    main()
