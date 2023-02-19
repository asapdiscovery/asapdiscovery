"""
Create oedu binary DesignUnit files for the given fragalysis structures. This
script assumes that there is a ligand bound, and that the ligand will be used
to dock against.
"""
import argparse
import multiprocessing as mp
import os
import re
import sys
from tempfile import NamedTemporaryFile

import yaml
from openeye import oechem

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asapdiscovery.data.fragalysis import parse_xtal  # noqa: E402
from asapdiscovery.data.openeye import load_openeye_pdb  # noqa: E402
from asapdiscovery.data.openeye import save_openeye_pdb  # noqa: E402
from asapdiscovery.data.utils import edit_pdb_file  # noqa: E402
from asapdiscovery.data.utils import seqres_to_res_list  # noqa: E402
from asapdiscovery.docking.modeling import du_to_complex  # noqa: E402
from asapdiscovery.docking.modeling import mutate_residues  # noqa: 402
from asapdiscovery.docking.modeling import prep_receptor  # noqa: E402


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


def prep_mp(xtal, seqres, out_base, loop_db):
    # Get chain
    re_pat = rf"/{xtal.dataset}_([0-9][A-Z])/"
    try:
        chain = re.search(re_pat, xtal.str_fn).groups()[0]
    except AttributeError:
        print(
            f"Regex chain search failed: {re_pat}, {str_fn}.",  # noqa: F821
            "Using 0A as default.",
            flush=True,
        )
        chain = "0A"

    # Check if results already exist
    out_dir = os.path.join(out_base, f"{xtal.dataset}_{chain}_{xtal.compound_id}")
    if check_completed(out_dir):
        return

    # Make output directory
    os.makedirs(out_dir, exist_ok=True)

    # Option to add SEQRES header
    if seqres:
        # Get a list of 3-letter codes for the sequence
        res_list = seqres_to_res_list(seqres)

        # Generate a new (temporary) pdb file with the SEQRES we want
        with NamedTemporaryFile(mode="w", suffix=".pdb") as tmp_pdb:
            # Add the SEQRES
            edit_pdb_file(xtal.str_fn, seqres_str=seqres, pdb_out=tmp_pdb.name)

            # Load in the pdb file as an OE object
            seqres_prot = load_openeye_pdb(tmp_pdb.name)

            # Mutate the residues to match the residue list
            initial_prot = mutate_residues(seqres_prot, res_list)
    else:
        initial_prot = load_openeye_pdb(xtal.str_fn)

    # Take the first returned DU and save it
    try:
        design_units = prep_receptor(
            initial_prot,
            loop_db=loop_db,
        )
    except IndexError:
        print(
            "DU generation failed for",
            f"{xtal.dataset}_{chain}_{xtal.compound_id})",
            flush=True,
        )
        return

    du = design_units[0]
    print(
        f"{xtal.dataset}_{chain}_{xtal.compound_id}",
        oechem.OEWriteDesignUnit(os.path.join(out_dir, "prepped_receptor.oedu"), du),
        flush=True,
    )

    # Save complex as PDB file
    complex_mol = du_to_complex(du, include_solvent=True)
    save_openeye_pdb(complex_mol, os.path.join(out_dir, "prepped_complex.pdb"))


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
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

    # Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    # Model-building arguments
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

    # Performance arguments
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

    mp_args = [(x, seqres, args.output_dir, args.loop_db) for x in xtal_compounds]
    print(mp_args[0], flush=True)
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
