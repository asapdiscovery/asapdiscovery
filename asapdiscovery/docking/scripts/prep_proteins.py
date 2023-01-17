"""
Create oedu binary DesignUnit files for input protein structures.
This was designed to be able to be used for either molecules from the PDB or from fragalysis.

Example Usage:
    python prep_proteins.py \
    -d /data/chodera/asap-datasets/mers_pdb_download \
    -p ../data/mers-structures-dimers.yaml \
    -r ~/fragalysis/extra_files/reference.pdb \
    -l ~/rcsb_spruce.loop_db \
    -o /data/chodera/asap-datasets/mers_prepped_structures \
    -s ../data/mpro_mers_seqres.yaml \
    --protein_only
"""
import argparse
import multiprocessing as mp
from openeye import oechem
import os
import re
import sys
from tempfile import NamedTemporaryFile
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.docking.modeling import (
    align_receptor,
    prep_receptor,
    du_to_complex,
    mutate_residues,
    remove_extra_ligands,
)
from asapdiscovery.data import pdb
from asapdiscovery.data.utils import edit_pdb_file, seqres_to_res_list
from asapdiscovery.data.openeye import save_openeye_pdb, load_openeye_pdb
from asapdiscovery.data.fragalysis import parse_xtal


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
        not os.path.isfile(os.path.join(d, "prepped_receptor.pdb"))
    ):
        return False

    try:
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(os.path.join(d, "prepped_receptor.oedu"), du)
    except Exception:
        return False

    try:
        _ = load_openeye_pdb(os.path.join(d, "prepped_receptor.pdb"))
    except Exception:
        return False

    return True


def prep_mp(
    xtal: CrystalCompoundData,
    ref_prot,
    seqres,
    out_base,
    loop_db,
    protein_only: bool,
):
    ## Check if results already exist
    out_dir = os.path.join(out_base, f"{xtal.output_name}")
    # if check_completed(out_dir):
    #     return

    ## Make output directory
    os.makedirs(out_dir, exist_ok=True)

    ## Load protein from pdb
    initial_prot = load_openeye_pdb(xtal.str_fn)

    if seqres:
        res_list = seqres_to_res_list(seqres)

        print("Mutating to provided seqres")
        ## Mutate the residues to match the residue list
        initial_prot = mutate_residues(
            initial_prot, res_list, xtal.protein_chains
        )

    ## Delete extra copies of ligand in the complex
    initial_prot = remove_extra_ligands(
        initial_prot, lig_chain=xtal.active_site_chain
    )

    if ref_prot:
        print("Aligning receptor")
        initial_prot = align_receptor(
            initial_complex=initial_prot,
            ref_prot=ref_prot,
            dimer=True,
            split_initial_complex=protein_only,
            mobile_chain=xtal.active_site_chain,
            ref_chain="A",
        )
        save_openeye_pdb(initial_prot, "align_test.pdb")
    ## Take the first returned DU and save it
    try:
        print("Attempting to prepare design units")
        site_residue = xtal.active_site if xtal.active_site else ""
        design_units = prep_receptor(
            initial_prot,
            site_residue=site_residue,
            loop_db=loop_db,
            protein_only=protein_only,
            seqres=" ".join(res_list),
        )
    except IndexError as e:
        print(
            "DU generation failed for",
            f"{xtal.output_name}",
            flush=True,
        )
        return

    du = design_units[0]
    for i, du in enumerate(design_units):
        print(
            f"{xtal.output_name}",
            oechem.OEWriteDesignUnit(
                os.path.join(out_dir, f"prepped_receptor_{i}.oedu"), du
            ),
            flush=True,
        )

        ## Save complex as PDB file
        complex_mol = du_to_complex(du, include_solvent=True)

        ## Add SEQRES entries if they're not present
        if not oechem.OEHasPDBData(complex_mol, "SEQRES"):
            for seqres_line in seqres.split("\n"):
                if seqres_line != "":
                    oechem.OEAddPDBData(complex_mol, "SEQRES", seqres_line[6:])

        save_openeye_pdb(
            complex_mol, os.path.join(out_dir, f"prepped_receptor_{i}.pdb")
        )


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
    parser.add_argument(
        "--protein_only",
        action="store_true",
        default=False,
        help="If true, generate design units with only the protein in them",
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
            ## The parentheses in this string are the capture group
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

            ## We also want the chain in the form of a single letter ('A', 'B'), etc
            xtal.active_site_chain = frag_chain[-1]

            ## If we aren't keeping the ligands, then we want to give it a site residue to use
            if args.protein_only:
                xtal.active_site = f"His:41: :{xtal.chain}"

    elif args.pdb_yaml_path:
        pdb_dict = pdb.load_pdbs_from_yaml(args.pdb_yaml_path)
        pdb_list = list(pdb_dict.keys())
        pdb.download_PDBs(pdb_list, args.structure_dir)
        active_site_chain = "A"
        ## If the yaml file doesn't have any options for the pdb file, then assume it is a dimer
        xtal_compounds = [
            CrystalCompoundData(
                str_fn=os.path.join(args.structure_dir, f"rcsb_{pdb_id}.pdb"),
                output_name=f"{pdb_id}_0{active_site_chain}",
                active_site_chain=active_site_chain,
                active_site=f"HIS:41: :{active_site_chain}",
                chains=values.get("chains", ["A", "B"]),
                oligomeric_state=values.get("oligomeric_state", "dimer"),
                protein_chains=values.get("protein_chains", ["A", "B"]),
            )
            for pdb_id, values in pdb_dict.items()
            if os.path.exists(
                os.path.join(args.structure_dir, f"rcsb_{pdb_id}.pdb")
            )
        ]

    if args.seqres_yaml:
        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]
    else:
        seqres = None

    mp_args = [
        (
            x,
            args.ref_prot,
            seqres,
            args.output_dir,
            args.loop_db,
            args.protein_only,
        )
        for x in xtal_compounds
    ]
    print(mp_args[0], flush=True)
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
