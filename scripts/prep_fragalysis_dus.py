"""
Create oedu binary DesignUnit files for the given fragalysis structures. This
script assumes that there is a ligand bound, and that the ligand will be used
to dock against.
"""
import argparse
import multiprocessing as mp
from openeye import oechem
import os
import pandas
import sys
from tempfile import NamedTemporaryFile
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covid_moonshot_ml.modeling import (
    align_receptor,
    prep_receptor,
    du_to_complex,
    mutate_residues,
)
from covid_moonshot_ml.datasets import pdb
from covid_moonshot_ml.datasets.utils import (
    save_openeye_pdb,
    split_openeye_mol,
    add_seqres,
    seqres_to_res_list,
    load_openeye_pdb,
)
from covid_moonshot_ml.docking.docking import parse_xtal


def prep_mp(xtal, seqres, ref_prot, out_base, chains, loop_db):
    ## Option to add SEQRES header
    if seqres:
        ## Get a list of 3-letter codes for the sequence
        res_list = seqres_to_res_list(seqres)

        ## Generate a new (temporary) pdb file with the SEQRES we want
        with NamedTemporaryFile(mode="w") as tmp_pdb:
            ## Add the SEQRES
            add_seqres(xtal.str_fn, seqres_str=seqres, pdb_out=tmp_pdb.name)

            ## Load in the pdb file as an OE object
            seqres_prot = load_openeye_pdb(tmp_pdb.name)

            ## Mutate the residues to match the residue list
            initial_prot = mutate_residues(seqres_prot, res_list)
    else:
        initial_prot = load_openeye_pdb(xtal.str_fn)

    for mobile_chain in chains:
        ## Make output directory
        out_dir = os.path.join(
            out_base, f"{xtal.dataset}_{xtal.compound_id}_{mobile_chain}"
        )
        os.makedirs(out_dir, exist_ok=True)

        aligned_prot = align_receptor(
            initial_complex=initial_prot,
            dimer=True,
            ref_prot=ref_prot,
            split_initial_complex=False,
            split_ref=False,
            ref_chain="A",
            mobile_chain=mobile_chain,
        )

        design_units = prep_receptor(
            aligned_prot,
            loop_db=loop_db,
        )

        ## Take the first returned DU and save it
        du = next(iter(design_units))
        oechem.OEWriteDesignUnit(
            os.path.join(out_dir, "prepped_receptor.oedu"), du
        )

        ## Save complex as PDB file
        complex_mol = du_to_complex(du, include_solvent=True)
        save_openeye_pdb(
            complex_mol, os.path.join(out_dir, "prepped_complex.pdb")
        )


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to fragalysis/aligned/ directory.",
    )
    parser.add_argument(
        "-x",
        "--xtal_csv",
        required=True,
        help="CSV file giving information of which structures to prep.",
    )
    parser.add_argument(
        "-r",
        "--ref_prot",
        required=True,
        help="Path to reference pdb to align to.",
    )

    ## Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    ## Model-building arguments
    parser.add_argument(
        "-l",
        "--loop_db",
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "-c",
        "--chains",
        nargs="+",
        default=["A", "B"],
        help="Which chains to align to reference.",
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


def main():
    args = get_args()

    xtal_compounds = parse_xtal(args.xtal_csv, args.structure_dir)

    if args.seqres_yaml:
        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]
    else:
        seqres = None

    mp_args = [
        (x, seqres, args.ref_prot, args.output_dir, args.chains, args.loop_db)
        for x in xtal_compounds
    ]
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Running {len(mp_args)} docking runs over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
