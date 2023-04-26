import argparse
import pickle as pkl
import shutil


from datetime import datetime
from pathlib import Path
from typing import List


from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.utils import (
    oe_load_exp_from_file,
    is_valid_smiles,
    exp_data_to_oe_mols,
)
from asapdiscovery.docking import prep_mp
from asapdiscovery.docking.mcs import rank_structures_openeye  # noqa: E402
from asapdiscovery.docking.mcs import rank_structures_rdkit  # noqa: E402
from asapdiscovery.docking.scripts.run_docking_oe import mp_func as oe_docking_function

# setup input arguments
parser = argparse.ArgumentParser(description="Run single target docking.")
parser.add_argument("-l", "--lig_file", help="SDF file containing ligands.")

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
    default="TARGET_MOL",
    help=(
        "Title of molecule to use if a SMILES string is passed in as input, default is to generate a new random UUID"
    ),
)

parser.add_argument(
    "-o",
    "--output_dir",
    required=True,
    help="Path to output_dir, will overwrite if exists.",
)

# general arguments
parser.add_argument(
    "--debug",
    action="store_true",
    help="enable debug mode, with more files saved and more verbose logging",
)

# general arguments
parser.add_argument(
    "--cleanup",
    action="store_true",
    help="clean up intermediate files",
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


# MCS arguments
parser.add_argument(
    "--mcs_sys",
    default="rdkit",
    help="Which package to use for MCS search [rdkit, oe].",
)

parser.add_argument(
    "--mcs_structural",
    action="store_true",
    help=("Use structure-based matching instead of element-based matching for MCS."),
)
parser.add_argument(
    "--n_draw",
    type=int,
    default=10,
    help="Number of MCS compounds to draw for each query molecule.",
)


# Docking arguments
parser.add_argument(
    "--top_n",
    type=int,
    default=1,
    help="Number of top matches to dock. Set to -1 to dock all.",
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
    "--num_poses",
    type=int,
    default=1,
    help="Number of poses to return from docking.",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Whether to print out verbose logging.",
)
parser.add_argument(
    "--gat",
    action="store_true",
    help="Whether to use GAT model to score docked poses.",
)


def main():
    args = parser.parse_args()

    # setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logname = "prep-mcs-dock-single-target"
    # setup logging
    logger = FileLogger(logname, path=output_dir, stdout=True).getLogger()
    logger.info(f"Start single target prep+docking at {datetime.now().isoformat()}")
    logger.info(f"Output directory: {output_dir}")

    # openeye logging handling
    errfs = oechem.oeofstream(output_dir / f"openeye-{logname}-log.txt")
    oechem.OEThrow.SetOutputStream(errfs)
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)

    if args.debug:
        logger.info("Running in debug mode.")
        logger.info(f"Input arguments: {args}")
        args.keep_intermediate = True

    # paths to remove if not keeping intermediate files
    intermediate_files = []

    # parse input molecules
    if is_valid_smiles(args.mols):
        logger.info(
            f"Input molecules is a single SMILES string: {args.mols}, using title {args.title}"
        )
        exp_data = [ExperimentalCompoundData(compound_id=args.title, smiles=args.mols)]
    else:
        mol_file_path = Path(args.mols)
        if mol_file_path.suffix == ".smi":
            logger.info(f"Input molecules is a SMILES file: {args.mols}")
            exp_data = oe_load_exp_from_file(args.mols, "smi")
        elif mol_file_path.suffix == ".sdf":
            logger.info(f"Input molecules is a SDF file: {args.mols}")
            exp_data = oe_load_exp_from_file(args.mols, "sdf")
        else:
            raise ValueError(
                f"Input molecules must be a SMILES file, SDF file, or SMILES string. Got {args.mols}"
            )

    logger.info(f"Loaded {len(exp_data)} molecules.")

    # parse prep arguments
    prep_dir = output_dir / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    intermediate_files.append(prep_dir)

    logger.info(f"Prepping receptor in {prep_dir} at {datetime.now().isoformat()}")

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

    # load receptor, may need to work on how to provide arguments to this
    # check with @jenke
    receptor_name = receptor.stem
    xtal = CrystalCompoundData(
        str_fn=args.receptor, smiles=None, output_name=str(receptor_name)
    )

    logger.info(f"Loaded receptor {receptor_name} from {receptor}")
    prep_mp(
        xtal, args.ref_prot, seqres, args.output_dir, args.loop_db, args.protein_only
    )
    logger.info(f"Finished prepping receptor at {datetime.now().isoformat()}")

    # grab the files that were created
    prepped_oedu = prep_dir / f"{receptor_name}_prepped_receptor_0.oedu"
    prepped_pdb = prep_dir / f"{receptor_name}_prepped_receptor_0.pdb"

    logger.info(f"Prepped receptor: {prepped_pdb}, {prepped_oedu}")

    # read design unit and split it
    du = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(prepped_oedu), du)

    lig, prot, complexed = split_openeye_design_unit(du, lig=True)

    if args.debug:
        # write out the ligand and protein
        save_openeye_sdf(str(prep_dir / f"{receptor_name}_ligand.sdf"), lig)
        save_openeye_pdb(str(prep_dir / f"{receptor_name}_protein.pdb"), prot)

    ligand_smiles = oechem.OEMolToSmiles(lig)

    logger.info(f"Xtal ligand: {ligand_smiles}")

    # setup MCS search
    if args.mcs_sys == "rdkit":
        logger.info(f"Using RDKit for MCS search.")
        mcs_rank_fn = rank_structures_rdkit
    elif args.mcs_sys == "oe":
        logger.info(f"Using OpenEye for MCS search.")
        mcs_rank_fn = rank_structures_openeye
    else:
        raise ValueError(f"Invalid MCS search system: {args.mcs_sys}")

    # run MCS search
    logger.info(f"Running MCS search at {datetime.now().isoformat()}")
    sort_idxs = []
    for compound in exp_data:
        sort_idxs.append(
            mcs_rank_fn(
                compound.smiles,
                c.compound_id,
                ligand_smiles,
                receptor_name,
                None,
                args.mcs_structural,
                None,
                args.n_draw,
            )
        )
    logger.info(f"Finished MCS search at {datetime.now().isoformat()}")
    if args.debug:
        logger.info(
            f"Saving MCS search results to {args.o}/mcs_sort_index.pkl for debugging."
        )
        compound_ids = [c.compound_id for c in exp_compounds]
        xtal_ids = [x.dataset for x in xtal_compounds]

        pkl.dump(
            [compound_ids, xtal_ids, sort_idxs],
            open(f"{args.o}/mcs_sort_index.pkl", "wb"),
        )

    # setup docking
    logger.info(f"Starting docking setup at {datetime.now().isoformat()}")

    dock_dir = output_dir / "docking"
    dock_dir.mkdir(parents=True, exist_ok=True)
    intermediate_files.append(dock_dir)

    # ML stuff for docking
    logger.info(f"Setup ML for docking")
    gat_model_string = "asapdiscovery-GAT-2023.04.12"
    e3nn_model_string = None
    schnet_model_string = None

    if args.gat:
        from asapdiscovery.ml.inference import GATInference  # noqa: E402

        gat_model = GATInference(gat_model_string)
        logger.info(f"Using GAT model: {gat_model_string}")
    else:
        logger.info("Skipping GAT model scoring")
        gat_model = None

    # run docking
    logger.info(f"Running docking at {datetime.now().isoformat()}")

    results = []
    for mol in exp_data:
        results.append(
            oe_docking_function(
                docking_dir,
            )
        )
    logger.info(f"Finished docking at {datetime.now().isoformat()}")

    logger.info(f"Finish single target prep+docking at {datetime.now().isoformat()}")

    if args.cleanup:
        if len(intermediate_files) > 0:
            logger.info(f"Removing intermediate files.")
            for path in intermediate_files:
                shutil.rmtree(path)
    else:
        logger.info(f"Keeping intermediate files.")


if __name__ == "__main__":
    main()
