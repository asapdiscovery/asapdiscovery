"""
Create oedu binary DesignUnit files for input protein structures. This
script assumes that there is a ligand bound, and that the ligand will be used
to dock against.
"""
import argparse
import multiprocessing as mp
from openeye import oechem
import os
import pandas
import re
import sys
from tempfile import NamedTemporaryFile
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covid_moonshot_ml.schema import CrystalCompoundData
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


def check_completed(d):
    """
    Check if this prep process has already been run successfully in the given
    directory.

    Parameters
    ----------
    d : str
        Directory to check.

    Returns
    -------
    bool
        True if both files exist and can be loaded, otherwise False.
    """

    if (not os.path.isfile(os.path.join(d, "prepped_receptor.oedu"))) or (
        not os.path.isfile(os.path.join(d, "prepped_complex.pdb"))
    ):
        return False

    try:
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(os.path.join(d, "prepped_receptor.oedu"), du)
    except Exception:
        return False

    try:
        _ = load_openeye_pdb(os.path.join(d, "prepped_complex.pdb"))
    except Exception:
        return False

    return True


def prep_mp(xtal: CrystalCompoundData, ref_prot, seqres, out_base, loop_db):
    ## Check if results already exist
    out_dir = os.path.join(out_base, f"{xtal.output_name}")
    if check_completed(out_dir):
        return

    ## Make output directory
    os.makedirs(out_dir, exist_ok=True)

    ## Option to add SEQRES header
    if seqres:
        ## Get a list of 3-letter codes for the sequence
        res_list = seqres_to_res_list(seqres)

        ## Generate a new (temporary) pdb file with the SEQRES we want
        with NamedTemporaryFile(mode="w", suffix=".pdb") as tmp_pdb:
            ## Add the SEQRES
            add_seqres(xtal.str_fn, seqres_str=seqres, pdb_out=tmp_pdb.name)

            ## Load in the pdb file as an OE object
            seqres_prot = load_openeye_pdb(tmp_pdb.name)

            ## Mutate the residues to match the residue list
            initial_prot = mutate_residues(seqres_prot, res_list)
    else:
        initial_prot = load_openeye_pdb(xtal.str_fn)

    split_initial_complex = True if not xtal.compound_id else False

    if ref_prot:
        initial_prot = align_receptor(
            initial_complex=initial_prot,
            ref_prot=ref_prot,
            dimer=True,
            split_initial_complex=split_initial_complex,
            mobile_chain=xtal.chain,
            ref_chain="A",
        )

        save_openeye_pdb(initial_prot, "test.pdb")

    ## Take the first returned DU and save it
    try:
        design_units = prep_receptor(
            initial_prot,
            site_residue=xtal.active_site,
            loop_db=loop_db,
        )
    except IndexError as e:
        print(
            "DU generation failed for",
            f"{xtal.output_name})",
            flush=True,
        )
        return

    du = design_units[0]
    print(
        f"{xtal.output_name}",
        oechem.OEWriteDesignUnit(
            os.path.join(out_dir, "prepped_receptor.oedu"), du
        ),
        flush=True,
    )

    ## Save complex as PDB file
    complex_mol = du_to_complex(du, include_solvent=True)
    save_openeye_pdb(complex_mol, os.path.join(out_dir, "prepped_complex.pdb"))


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to fragalysis/aligned/ directory or directory to put PDB structures.",
    )

    parser.add_argument(
        "-x",
        "--xtal_csv",
        default=None,
        help="CSV file giving information of which structures to prep.",
    )
    parser.add_argument(
        "-p",
        "--pdb_yaml_path",
        default=None,
        help="Yaml file containing PDB IDs.",
    )

    parser.add_argument(
        "-r",
        "--ref_prot",
        default=None,
        type=str,
        help="Path to reference pdb to align to. If None, no alignment will be performed",
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

    if args.xtal_csv:
        xtal_compounds = parse_xtal(args.xtal_csv, args.structure_dir)

        for xtal in xtal_compounds:
            ## Get chain
            re_pat = rf"/{xtal.dataset}_([0-9][A-Z])/"
            try:
                frag_chain = re.search(re_pat, xtal.str_fn).groups()[0]
            except AttributeError:
                print(
                    f"Regex chain search failed: {re_pat}, {xtal.str_fn}.",
                    "Using A as default.",
                    flush=True,
                )
                frag_chain = "0A"
            xtal.output_name = f"{xtal.dataset}_{frag_chain}_{xtal.compound_id}"
            xtal.chain = frag_chain[-1]
            print(xtal.chain, xtal.output_name)
        xtal_compounds = [xtal for xtal in xtal_compounds if xtal.chain == "B"]

    elif args.pdb_yaml_path:
        pdb_list = pdb.load_pdbs_from_yaml(args.pdb_yaml_path)
        pdb.download_PDBs(pdb_list, args.structure_dir)
        pdb_fns = os.listdir(args.structure_dir)
        data = []
        for i, pdb_id in enumerate(pdb_list):
            data.append(
                (
                    os.path.join(args.structure_dir, pdb_fns[i]),
                    f"{pdb_id}_0A",
                    "A",
                    "HIS:41: :A",
                )
            )

        xtal_compounds = [
            CrystalCompoundData(
                str_fn=pdb_path,
                output_name=output_name,
                chain=chain,
                active_site=active_site,
            )
            for pdb_path, output_name, chain, active_site in data
            if os.path.exists(pdb_path)
        ]

    if args.seqres_yaml:
        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]
    else:
        seqres = None

    mp_args = [
        (x, args.ref_prot, seqres, args.output_dir, args.loop_db)
        for x in xtal_compounds
    ]
    mp_args = mp_args[0:1]
    print(mp_args[0], flush=True)
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
