import argparse
import datetime
import uuid
import shutil

from pathlib import Path
from typing import List

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.utils import oe_load_exp_from_file, is_valid_smiles



# setup input arguments
parser = argparse.ArgumentParser(description="Run single target docking.")
parser.add_argument("-l", "--lig_file", help="SDF file containing ligands.")

parser.add_argument(
    "-r",
    "--receptor",
    required=True,
    help=("Path to receptor to prep and dock to")
)

parser.add_argument(
    "-m",
    "--mols",
    required=True,
    help=("Path to the molecules to dock to the receptor as an SDF or SMILES file, or SMILES string.")
)

parser.add_argument(
    "--title",
    default="TARGET_MOL_" + str(uuid.uuid4()),
    help=("Title of molecule to use if a SMILES string is passed in as input, default is to generate a new random UUID")
)

parser.add_argument(
    "-o",
    "--output_dir",
    required=True,
    help="Path to output_dir, will overwrite if exists.",
)

parser.add_argument(
    "--keep_intermediate",
    required=True,
    action="store_true",
    help="Whether to keep intermediate files.",
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
    help=(
        "Use structure-based matching instead of element-based matching for MCS."
    )
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

parser.add_argument(
    "--e3nn",
    action="store_true",
    help="Whether to use e3nn model to score docked poses.",
)

parser.add_argument(
    "--schnet",
    action="store_true",
    help="Whether to use schnet model to score docked poses.",
)



def docking_func():
    pass


def main():
    args = parser.parse_args()

    # setup output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # setup logging
    log_name = args.log_name
    logger = FileLogger("single_target_workflow", path=output_dir).getLogger()
    logger.info(f"Start single target prep+docking at {datetime.now().isoformat()}")
    logger.info(f"Output directory: {output_dir}")


    # paths to remove if not keeping intermediate files
    intermediate_files = []


    # parse input molecules
    if is_valid_smiles(args.mols):
        logger.info(f"Input molecules is a single SMILES string: {args.mols}, using title {args.title}")
        exp_data = [ExperimentalCompoundData(compound_id=args.title, smiles=args.mols)]
    else:
        mol_file_path = Path(args.mols)
        if mol_file_path.suffix == ".smi":
            logger.info(f"Input molecules is a SMILES file: {args.mols}")
            exp_data = oe_load_exp_from_file(args.mols)
        elif mol_file_path.suffix == ".sdf":
            logger.info(f"Input molecules is a SDF file: {args.mols}")
            exp_data = oe_load_exp_from_file(args.mols)
        else:
            raise ValueError(
                f"Input molecules must be a SMILES file, SDF file, or SMILES string. Got {args.mols}"
            )
    
    logger.info(f"Loaded {len(exp_data)} molecules.")



    # setup receptor


    # prep receptor
    logger.info(f"Prepping receptor at {datetime.now().isoformat()}")
    prep_receptor(args.receptor)
    logger.info(f"Finished prepping receptor at {datetime.now().isoformat()}")

    





    
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
    mcs_rank_fn()
    logger.info(f"Finished MCS search at {datetime.now().isoformat()}")

    




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

    if args.e3nn:
        logger.warning("e3nn model not implemented yet, skipping")
        e3nn_model = None

    if args.schnet:
        logger.warning("schnet model not implemented yet, skipping")
        schnet_model = None

    full_docking_func = partial(
        docking_func,
        GAT_model=gat_model,
        e3nn_model=e3nn_model,
        schnet_model=schnet_model,
    )


    # run docking
    logger.info(f"Running docking at {datetime.now().isoformat()}")
    full_docking_func()
    logger.info(f"Finished docking at {datetime.now().isoformat()}")
    




    logger.info(f"Finish single target prep+docking at {datetime.now().isoformat()}")

    if args.keep_intermediate:
        logger.info(f"Keeping intermediate files.")
    else:
        if len(intermediate_files) > 0:
            logger.info(f"Removing intermediate files.")
            for path in intermediate_files:
                shutil.rmtree(path)


if __name__ == "__main__":
   main()

