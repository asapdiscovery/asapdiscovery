"""
Create oedu binary DesignUnit files for input protein structures.
This was designed to be able to be used for either molecules from the PDB or from fragalysis.

Example Usage:
    python prep_proteins.py \
    -d /Users/alexpayne/lilac-mount-point/asap-datasets/mers_pdb_download \
    -p ../../../metadata/mers-structures-dimers.yaml \
    -r /Users/alexpayne/lilac-mount-point/asap-datasets/mpro_fragalysis_2022_10_12/extra_files/reference.pdb \
    -l /Users/alexpayne/lilac-mount-point/asap-datasets/rcsb_spruce.loop_db \
    -o /Users/alexpayne/lilac-mount-point/asap-datasets/mers_fauxalysis/mers_prepped_structures_dimers_only \
    -s ../../../metadata/mpro_mers_seqres.yaml \
    --protein_only
"""
import argparse
import datetime
import multiprocessing as mp
from openeye import oechem
import os
import re
import sys
from tempfile import NamedTemporaryFile
import yaml
import logging

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
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
from asapdiscovery.data.openeye import (
    save_openeye_pdb,
    load_openeye_pdb,
    openeye_copy_pdb_data,
)
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

    if (not os.path.isfile(os.path.join(d, "prepped_receptor_0.oedu"))) or (
        not os.path.isfile(os.path.join(d, "prepped_receptor_0.pdb"))
    ):
        return False

    try:
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(os.path.join(d, "prepped_receptor_0.oedu"), du)
    except Exception:
        return False

    try:
        _ = load_openeye_pdb(os.path.join(d, "prepped_receptor_0.pdb"))
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
    handler = logging.FileHandler(os.path.join(out_dir, "log.txt"), mode="w")
    prep_logger = logging.getLogger(xtal.output_name)
    prep_logger.setLevel(logging.INFO)
    prep_logger.addHandler(handler)
    prep_logger.info(datetime.datetime.isoformat(datetime.datetime.now()))

    if check_completed(out_dir):
        prep_logger.info("Already completed! Finishing.")
        return

    ## Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # logging.basicConfig(
    #     filename=os.path.join(out_dir, "log.txt"),
    #     level=logging.DEBUG,
    #     filemode="w",
    # )
    prep_logger.info(f"Prepping {xtal.output_name}")

    ## Option to add SEQRES header
    if seqres:
        prep_logger.info("Editing PDB file")
        ## Get a list of 3-letter codes for the sequence
        res_list = seqres_to_res_list(seqres)

        ## Generate a new (temporary) pdb file with the SEQRES we want
        with NamedTemporaryFile(mode="w", suffix=".pdb") as tmp_pdb:
            ## Add the SEQRES
            edit_pdb_file(
                xtal.str_fn,
                seqres_str=seqres,
                edit_remark350=True,
                oligomeric_state=xtal.oligomeric_state,
                chains=xtal.chains,
                pdb_out=tmp_pdb.name,
            )

            ## Load in the pdb file as an OE object
            seqres_prot = load_openeye_pdb(tmp_pdb.name)

            save_openeye_pdb(seqres_prot, "seqres_test.pdb")

            initial_prot = seqres_prot
        mutate = True
    else:
        initial_prot = load_openeye_pdb(xtal.str_fn)
        mutate = False

    if mutate:
        prep_logger.info("Mutating to provided seqres")
        ## Mutate the residues to match the residue list
        initial_prot = mutate_residues(
            initial_prot, res_list, xtal.protein_chains
        )

    ## Delete extra copies of ligand in the complex
    initial_prot = remove_extra_ligands(
        initial_prot, lig_chain=xtal.active_site_chain
    )

    if ref_prot:
        prep_logger.info("Aligning receptor")
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
        prep_logger.info("Attempting to prepare design units")
        site_residue = xtal.active_site if xtal.active_site else ""
        design_units = prep_receptor(
            initial_prot,
            site_residue=site_residue,
            loop_db=loop_db,
            protein_only=protein_only,
        )
    except IndexError as e:
        prep_logger.error(
            "DU generation failed for",
            f"{xtal.output_name}",
        )
        return

    du = design_units[0]
    for i, du in enumerate(design_units):
        prep_logger.info(
            f"{xtal.output_name}",
            oechem.OEWriteDesignUnit(
                os.path.join(out_dir, f"prepped_receptor_{i}.oedu"), du
            ),
        )

        ## Save complex as PDB file
        complex_mol = du_to_complex(du, include_solvent=True)
        openeye_copy_pdb_data(complex_mol, initial_prot, "SEQRES")
        save_openeye_pdb(
            complex_mol, os.path.join(out_dir, f"prepped_receptor_{i}.pdb")
        )
    prep_logger.info(
        f"Finished protein prep at {datetime.datetime.isoformat(datetime.datetime.now())}"
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
    parser.add_argument(
        "--include_non_Pseries",
        default=False,
        action="store_true",
        help="If true, the p_only flag of parse_xtal will be set to False. Default is False, which sets p_only to True",
    )
    parser.add_argument(
        "--log_file",
        default="prep_proteins_log.txt",
        help="Path to high level log file.",
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

    # logging.basicConfig(
    #     filename=args.log_file,
    #     level=logging.DEBUG,
    #     filemode="w",
    # )
    handler = logging.FileHandler(args.log_file, mode="w")
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(handler)

    if args.xtal_csv:
        p_only = False if args.include_non_Pseries else True
        xtal_compounds = parse_xtal(
            args.xtal_csv, args.structure_dir, p_only=p_only
        )

        for xtal in xtal_compounds:
            ## Get chain
            ## The parentheses in this string are the capture group
            re_pat = rf"/{xtal.dataset}_([0-9][A-Z])/"
            try:
                frag_chain = re.search(re_pat, xtal.str_fn).groups()[0]
            except AttributeError:
                main_logger.error(
                    f"Regex chain search failed: {re_pat}, {xtal.str_fn}.",
                    "Using A as default.",
                )
                frag_chain = "0A"
            xtal.output_name = f"{xtal.dataset}_{frag_chain}_{xtal.compound_id}"

            ## We also want the chain in the form of a single letter ('A', 'B'), etc
            xtal.active_site_chain = frag_chain[-1]

            ## If we aren't keeping the ligands, then we want to give it a site residue to use
            if args.protein_only:
                xtal.active_site = f"His:41: :{xtal.chain}"

    elif args.pdb_yaml_path:
        ## First, get list of pdbs from yaml file
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
    else:
        raise NotImplementedError("Crystal CSV or PDB yaml file needed")

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
    main_logger.info(mp_args[0])
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    main_logger.info(
        f"CPUS: {mp.cpu_count()}, Structure: {mp_args}, N Cores: {args.num_cores}"
    )
    main_logger.info(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
