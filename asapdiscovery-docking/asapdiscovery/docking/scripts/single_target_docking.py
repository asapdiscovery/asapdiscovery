import argparse
import hashlib
import logging
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path  # noqa: F401
from typing import List  # noqa: F401

import yaml
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    extract_ligand_from_design_unit,
    oechem,
    save_openeye_sdf,
)
from asapdiscovery.data.schema import CrystalCompoundData, ExperimentalCompoundData
from asapdiscovery.data.utils import (
    exp_data_to_oe_mols,
    is_valid_smiles,
    oe_load_exp_from_file,
)
from asapdiscovery.dataviz.html_vis import HTMLVisualiser
from asapdiscovery.docking import make_docking_result_dataframe
from asapdiscovery.docking import prep_mp as oe_prep_function
from asapdiscovery.docking.mcs import rank_structures_openeye  # noqa: F401
from asapdiscovery.docking.mcs import rank_structures_rdkit  # noqa: F401
from asapdiscovery.docking.scripts.run_docking_oe import mp_func as oe_docking_function
from asapdiscovery.simulation.simulate import VanillaMDSimulator
from rdkit import Chem

"""
Script to run single target prep + docking.

Input:
    - receptor: path to receptor to prep and dock to
    - mols: path to the molecules to dock to the receptor as an SDF or SMILES file, or SMILES string.
    - title: title of molecule to use if a SMILES string is passed in as input, default is to hash the SMILES string to avoid accidental caching.
    - output_dir: path to output_dir, will NOT overwrite if exists.
    - debug: enable debug mode, with more files saved and more verbose logging
    - verbose: whether to print out verbose logging.
    - cleanup: clean up intermediate files

    # Prep arguments
    - loop_db: path to loop database.
    - seqres_yaml: path to yaml file of SEQRES.
    - protein_only: if true, generate design units with only the protein in them
    - ref_prot: path to reference pdb to align to. If None, no alignment will be performed

    # MCS arguments
    - mcs_sys: which package to use for MCS search [rdkit, oe].
    - mcs_structural: use structure-based matching instead of element-based matching for MCS.
    - n_draw: number of MCS compounds to draw for each query molecule.

    # Docking arguments
    - docking_sys: which docking system to use [posit, hybrid]. Defaults to posit.
    - relax: when to run relaxation [none, clash, all]. Defaults to none.
    - omega: use Omega conformer enumeration.
    - hybrid: whether to only use hybrid docking protocol in POSIT.
    - num_poses: number of poses to return from docking.
    - gat: whether to use GAT model to score docked poses.


Example usage:

    # with an SDF or SMILES file

    python single_target_docking.py \
        -r /path/to/receptor.pdb \
        -m /path/to/mols.[sdf/smi] \
        -o /path/to/output_dir \

    # with a SMILES string

    python single_target_docking.py \
        -r /path/to/receptor.pdb \
        -m 'COC(=O)COc1cc(cc2c1OCC[C@H]2C(=O)Nc3cncc4c3cccc4)Cl'
        -o /path/to/output_dir \
        --title 'my_fancy_molecule' \\ # without a title you will be given a hash of the SMILES string
        --debug \

"""


# setup input arguments
parser = argparse.ArgumentParser(description="Run single target docking.")

parser.add_argument(
    "-r", "--receptor", required=True, help=("Path to receptor to prep and dock to")
)

parser.add_argument(
    "-m",
    "--mols",
    required=True,
    help=(
        "Path to the molecules to dock to the receptor as an SDF or SMILES file, or SMILES string."
    ),
)

parser.add_argument(
    "--title",
    default=None,
    type=str,
    help=(
        "Title of molecule to use if a SMILES string is passed in as input, default is to use the SMILES string."
    ),
)

parser.add_argument(
    "--smiles_as_title",
    action="store_true",
    help=(
        "use smiles strings as titles for molecules in .smi or .sdf file if none provided"
    ),
)

parser.add_argument(
    "-o",
    "--output_dir",
    required=True,
    help="Path to output_dir, will NOT overwrite if exists.",
)

# general arguments
parser.add_argument(
    "--debug",
    action="store_true",
    help="enable debug mode, with more files saved and more verbose logging",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Whether to print out verbose logging.",
)

# general arguments
parser.add_argument(
    "--cleanup",
    action="store_true",
    help="clean up intermediate files",
)

parser.add_argument(
    "--logname",
    default="single_target_docking",
    type=str,
    help="Name of log file",
)

# Prep arguments
parser.add_argument(
    "--loop_db",
    help="Path to loop database.",
)
parser.add_argument(
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
    "--ref_prot",
    default=None,
    type=str,
    help="Path to reference pdb to align to. If None, no alignment will be performed",
)

# Docking arguments
parser.add_argument(
    "--use_3d",
    action="store_true",
    help="Whether to use 3D coordinates from SDF input file for docking, default is to use 2D representation.",
)

parser.add_argument(
    "--docking_sys",
    default="posit",
    help="Which docking system to use [posit, hybrid]. Defaults to posit.",
)
parser.add_argument(
    "--relax",
    default="none",
    help="When to run relaxation [none, clash, all]. Defaults to none.",
)

parser.add_argument(
    "--omega",
    action="store_true",
    help="Use Omega conformer enumeration.",
)

parser.add_argument(
    "--hybrid",
    action="store_true",
    help="Whether to only use hybrid docking protocol in POSIT.",
)

parser.add_argument(
    "--num_poses",
    type=int,
    default=1,
    help="Number of poses to return from docking.",
)

parser.add_argument(
    "--gat",
    action="store_true",
    help="Whether to use GAT model to score docked poses.",
)


parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="Target to write visualisations for, one of (sars2, mers, 7ene)",
)

parser.add_argument(
    "--md", action="store_true", help="Whether to run MD after docking."
)


def main():
    args = parser.parse_args()

    # setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logname = args.logname if args.logname else "single_target_docking"
    # setup logging
    logger_cls = FileLogger(logname, path=output_dir, stdout=True)
    logger = logger_cls.getLogger()
    logger.info(f"Start single target prep+docking at {datetime.now().isoformat()}")
    logger.info(f"Output directory: {output_dir}")

    # openeye logging handling
    errfs = oechem.oeofstream(str(output_dir / f"openeye-{logname}-log.txt"))
    oechem.OEThrow.SetOutputStream(errfs)
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)

    if args.debug:
        logger.info("Running in debug mode. enabling --verbose and disabling --cleanup")
        args.verbose = True
        args.cleanup = False

    if args.verbose:
        logger_cls.set_level(logging.DEBUG)
        logger = logger_cls.getLogger()
        logger.debug("Debug logging enabled")
        logger.debug(f"Input arguments: {args}")

    # paths to remove if not keeping intermediate files
    intermediate_files = []

    # set internal flag to seed whether we used 3D or 2D chemistry input
    used_3d = False
    # parse input molecules
    if is_valid_smiles(args.mols):
        logger.info(
            f"Input molecules is a single SMILES string: {args.mols}, using title {args.title}"
        )
        # hash the smiles to generate a unique title and avoid accidentally caching different outputs as the same
        if not args.title:
            logger.info(
                "No title provided, using SMILES string to generate title, consider providing a title with --title"
            )
            args.title = "TARGET_MOL-" + args.mols

        exp_data = [ExperimentalCompoundData(compound_id=args.title, smiles=args.mols)]
    else:
        mol_file_path = Path(args.mols)
        if mol_file_path.suffix == ".smi":
            logger.info(f"Input molecules is a SMILES file: {args.mols}")
            exp_data = oe_load_exp_from_file(
                args.mols, "smi", smiles_as_title=args.smiles_as_title
            )
        elif mol_file_path.suffix == ".sdf":
            logger.info(f"Input molecules is a SDF file: {args.mols}")
            if args.use_3d:
                logger.info("Using 3D coordinates from SDF file")
                # we need to keep the molecules around to retain their coordinates
                exp_data, oe_mols = oe_load_exp_from_file(
                    args.mols,
                    "sdf",
                    return_mols=True,
                    smiles_as_title=args.smiles_as_title,
                )
                used_3d = True
                logger.info("setting used_3d to True")
            else:
                logger.info("Using 2D representation from SDF file")
                exp_data = oe_load_exp_from_file(
                    args.mols, "sdf", smiles_as_title=args.smiles_as_title
                )

        else:
            raise ValueError(
                f"Input molecules must be a SMILES file, SDF file, or SMILES string. Got {args.mols}"
            )

    logger.info(f"Loaded {len(exp_data)} molecules.")

    if args.verbose:
        # we could make this just a debug statement but avoid looping over all molecules if not needed
        for exp in exp_data:
            logger.debug(f"Loaded molecule {exp.compound_id}: {exp.smiles}")

    # parse prep arguments
    prep_dir = output_dir / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    intermediate_files.append(prep_dir)

    logger.info(f"Prepping receptor in {prep_dir} at {datetime.now().isoformat()}")

    # check inputs
    receptor = Path(args.receptor)
    if receptor.suffix != ".pdb":
        raise ValueError(f"Receptor must be a PDB file. Got {args.receptor}")

    if not receptor.exists():
        raise ValueError(f"Receptor file does not exist: {args.receptor}")

    if args.ref_prot is not None:
        ref_prot = Path(args.ref_prot)
        if ref_prot.suffix != ".pdb":
            raise ValueError(
                f"Reference protein must be a PDB file. Got {args.ref_prot}"
            )

    if args.loop_db is not None:
        loop_db = Path(args.loop_db)
        if not loop_db.exists():
            raise ValueError(f"Loop database file does not exist: {args.loop_db}")

    if args.seqres_yaml is not None:
        seqres_yaml = Path(args.seqres_yaml)
        # check it exists
        if not seqres_yaml.exists():
            raise ValueError(f"SEQRES yaml file does not exist: {args.seqres_yaml}")

        # load it
        logger.info(f"Using SEQRES from {args.seqres_yaml}")
        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]
    else:
        seqres = None

    # load the receptor
    receptor_name = receptor.stem
    xtal = CrystalCompoundData(
        str_fn=args.receptor, smiles=None, output_name=str(receptor_name)
    )

    logger.info(f"Loaded receptor {receptor_name} from {receptor}")
    oe_prep_function(
        xtal, args.ref_prot, seqres, prep_dir, args.loop_db, args.protein_only
    )
    logger.info(f"Finished prepping receptor at {datetime.now().isoformat()}")

    # grab the files that were created
    prepped_oedu = (
        prep_dir / f"{receptor_name}" / f"{receptor_name}_prepped_receptor_0.oedu"
    )
    prepped_pdb = (
        prep_dir / f"{receptor_name}" / f"{receptor_name}_prepped_receptor_0.pdb"
    )

    # check the prepped receptors exist
    if not prepped_oedu.exists() or not prepped_pdb.exists():
        raise ValueError(
            f"Prepped receptor files do not exist: {prepped_oedu}, {prepped_pdb}"
        )

    logger.info(f"Prepped receptor: {prepped_pdb}, {prepped_oedu}")

    # read design unit and split it
    du = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(prepped_oedu), du)

    # extract the ligand
    lig = extract_ligand_from_design_unit(du)

    if args.debug:
        # write out the ligand and protein
        logger.info("Writing out ligand for debugging")
        save_openeye_sdf(lig, str(prep_dir / f"{receptor_name}_ligand.sdf"))

    ligand_smiles = oechem.OEMolToSmiles(lig)

    logger.info(f"Xtal ligand: {ligand_smiles}")

    # setup docking
    logger.info(f"Starting docking setup at {datetime.now().isoformat()}")

    dock_dir = output_dir / "docking"
    dock_dir.mkdir(parents=True, exist_ok=True)
    intermediate_files.append(dock_dir)

    # ML stuff for docking, fill out others as we make them
    logger.info("Setup ML for docking")
    gat_model_string = "asapdiscovery-GAT-2023.04.12"

    if args.gat:
        from asapdiscovery.ml.inference import GATInference  # noqa: E402

        gat_model = GATInference(gat_model_string)
        logger.info(f"Using GAT model: {gat_model_string}")
    else:
        logger.info("Skipping GAT model scoring")
        gat_model = None

    # use partial to bind the ML models to the docking function
    full_oe_docking_function = partial(oe_docking_function, GAT_model=gat_model)

    # run docking
    logger.info(f"Running docking at {datetime.now().isoformat()}")

    results = []

    # if we havn't already made the OE representation of the molecules, do it now
    if not used_3d:
        logger.info("Loading molecules from SMILES")
        oe_mols = exp_data_to_oe_mols(exp_data)
    else:
        logger.info("Using 3D molecules from input")

    for mol, compound in zip(oe_mols, exp_data):
        logger.debug(f"Running docking for {compound.compound_id}")
        if args.debug:
            # check smiles match
            if compound.smiles != oechem.OEMolToSmiles(mol):
                raise ValueError(
                    f"SMILES mismatch between {compound.compound_id} and {mol.GetTitle()}"
                )
        results.append(
            full_oe_docking_function(
                dock_dir / f"{compound.compound_id}_{receptor_name}",
                compound.compound_id,
                prepped_oedu,
                logname,
                f"{compound.compound_id}_{receptor_name}",
                du,
                mol,
                args.docking_sys.lower(),
                args.relax.lower(),
                args.hybrid,
                f"{compound.compound_id}_{receptor_name}",
                args.omega,
                args.num_poses,
            )
        )
    logger.info(f"Finished docking at {datetime.now().isoformat()}")
    logger.info(f"Docking finished for {len(results)} runs.")

    # save results
    results_df, csv = make_docking_result_dataframe(results, output_dir, save_csv=True)
    logger.info(f"Saved results to {csv}")
    logger.info(f"Finish single target prep+docking at {datetime.now().isoformat()}")

    poses_dir = output_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    intermediate_files.append(poses_dir)
    logging.info(f"Starting docking result processing at {datetime.now().isoformat()}")
    logger.info(f"Processing {len(results_df)} docking results")

    # only write out visualisation for the best posit score for each ligand
    # sort by posit  score
    sorted_df = results_df.sort_values(by=["POSIT_prob"], ascending=False)
    top_posit = sorted_df.drop_duplicates(subset=["ligand_id"], keep="first")
    top_posit.to_csv(output_dir / "top_poses.csv", index=False)

    logger.info(
        f"Writing out visualisation for top pose for each ligand (n={len(top_posit)})"
    )

    # add pose output column
    top_posit["outpath_pose"] = top_posit["ligand_id"].apply(
        lambda x: poses_dir / Path(x) / "visualisation.html"
    )

    html_visualiser = HTMLVisualiser(
        top_posit["docked_file"], top_posit["outpath_pose"], args.target, prepped_pdb
    )
    html_visualiser.write_pose_visualisations()

    if args.md:
        logger.info(f"Running MD on top pose for each ligand (n={len(top_posit)})")
        md_dir = output_dir / "md"
        md_dir.mkdir(parents=True, exist_ok=True)
        intermediate_files.append(md_dir)

        top_posit["outpath_md"] = top_posit["ligand_id"].apply(
            lambda x: md_dir / Path(x)
        )

        logger.info(f"Starting MD at {datetime.now().isoformat()}")
        # simulator = VanillaMDSimulator(top_posit["docked_file"], prepped_pdb, logger=logger, output_paths=top_posit["outpath_md"])
        # simulator.run_simulations()
        logger.info(f"Finished MD at {datetime.now().isoformat()}")

    if args.cleanup:
        if len(intermediate_files) > 0:
            logger.info("Removing intermediate files.")
            for path in intermediate_files:
                shutil.rmtree(path)
    else:
        logger.info("Keeping intermediate files.")


if __name__ == "__main__":
    main()
