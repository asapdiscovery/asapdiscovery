import argparse
import logging
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path  # noqa: F401
from typing import List  # noqa: F401

import dask
from asapdiscovery.data.execution_utils import get_interfaces_with_dual_ip
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import load_openeye_design_unit, oechem
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.schema import CrystalCompoundData, ExperimentalCompoundData
from asapdiscovery.data.utils import (
    exp_data_to_oe_mols,
    is_valid_smiles,
    oe_load_exp_from_file,
)
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from asapdiscovery.dataviz.viz_targets import VizTargets
from asapdiscovery.docking import (
    POSIT_METHODS,
    dock_and_score_pose_oe,
    make_docking_result_dataframe,
)
from asapdiscovery.docking.docking_data_validation import (
    drop_and_rename_docking_output_cols_for_manifold,
)
from asapdiscovery.modeling.modeling import protein_prep_workflow
from asapdiscovery.modeling.schema import (
    MoleculeFilter,
    PrepOpts,
    PreppedTarget,
    PreppedTargets,
)
from asapdiscovery.simulation.simulate import VanillaMDSimulator

"""
Script to run single target prep + docking.

Input:
    - receptor: path to receptor to prep and dock to
    - mols: path to the molecules to dock to the receptor as an SDF or SMILES file, or SMILES string.
    - title: title of molecule to use if a SMILES string is passed in as input, default is to use an index
    - smiles_as_title: use smiles strings as titles for molecules in .smi or .sdf file if none provided
    - dask: use dask to parallelise docking
    - dask-lilac: run dask in lilac config mode
    - output_dir: path to output_dir, will overwrite if exists.
    - debug: enable debug mode, with more files saved and more verbose logging
    - verbose: whether to print out verbose logging.
    - cleanup: clean up intermediate files except for final csv and visualizations
    - logname: name of logger

    # Prep arguments
    - loop_db: path to loop database.
    - seqres_yaml: path to yaml file of SEQRES.
    - protein_only: if true, generate design units with only the protein in them
    - ref_prot: path to reference pdb to align to. If None, no alignment will be performed

    # Docking arguments
    - docking_sys: which docking system to use [posit, hybrid]. Defaults to posit.
    - relax: when to run relaxation [none, clash, all]. Defaults to none.
    - omega: use Omega conformer enumeration.
    - hybrid: whether to only use hybrid docking protocol in POSIT.
    - num_poses: number of poses to return from docking.
    - gat: whether to use GAT model to score docked poses.


    # MD arguments
    - md: whether to run MD after docking.
    - md-steps: number of MD steps to run, default 2500000 for a 10 ns simulation at 4 fs timestep.

Example usage:

    # with an SDF or SMILES file

    dock-small-scale-e2e \
        -r /path/to/receptor.pdb \
        -m /path/to/mols.[sdf/smi] \
        -o /path/to/output_dir \

    # with a SMILES string using dask and requesting MD

    dock-small-scale-e2e \
        -r /path/to/receptor.pdb \
        -m 'COC(=O)COc1cc(cc2c1OCC[C@H]2C(=O)Nc3cncc4c3cccc4)Cl'
        -o /path/to/output_dir \
        --title 'my_fancy_molecule' \
        --debug \
        --dask \
        --md

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
        "Path to the molecules to dock to the receptor as an SDF or SMILES file, or SMILES string, or Postera MoleculeSet ID or Postera MoleculeSet name if --postera flag is set"
    ),
)

parser.add_argument(
    "-p",
    "--postera",
    action="store_true",
    help=(
        "Indicate that the input to the -m flag is a PostEra MoleculeSet ID or name, requires POSTERA_API_KEY environment variable to be set."
    ),
)

parser.add_argument(
    "--postera-upload",
    action="store_true",
    help=(
        "Indicates that the results of this docking run should be uploaded to PostEra, requires POSTERA_API_KEY environment variable to be set."
    ),
)

parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="Target protein name",
    choices=TargetTags.get_values(),
)

parser.add_argument(
    "--title",
    default=None,
    type=str,
    help=(
        "Title of molecule to use if a SMILES string is passed in as input, default is to use an index of the form unknown_lig_idx_<i>"
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
    "--dask",
    action="store_true",
    help=("use dask to parallelize docking"),
)

parser.add_argument(
    "--dask-lilac",
    action="store_true",
    help=("Run dask for lilac config"),
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
    "--ref_prot",
    default=None,
    type=str,
    help="Path to reference pdb to align to. If None, no alignment will be performed",
)

parser.add_argument(
    "--components_to_keep", type=str, nargs="+", default=["protein", "ligand"]
)
parser.add_argument("--active_site_chain", type=str, default="A")

parser.add_argument("--ligand_chain", type=str, default="A")

parser.add_argument("--protein_chains", type=str, default=[], help="")

parser.add_argument(
    "--oe_active_site_residue",
    type=str,
    default=None,
    help="OpenEye formatted site residue for active site identification otherwise will use OpenEye automatic detection, i.e. 'HIS:41: :A:0: '",
)

parser.add_argument(
    "--ref_chain", type=str, default="A", help="Chain of reference to align to."
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
    default="clash",
    help="When to run relaxation [none, clash, all]. Defaults to clash.",
)

parser.add_argument(
    "--no-omega",
    action="store_true",
    help="Do not use Omega conformer enumeration.",
)
parser.add_argument(
    "--posit_method",
    type=str,
    default="all",
    choices=POSIT_METHODS,
    help="Which POSIT method to use for POSIT docking protocol.",
)

parser.add_argument(
    "--num_poses",
    type=int,
    default=1,
    help="Number of poses to return from docking.",
)


# ML arguments
parser.add_argument(
    "--no-gat",
    action="store_true",
    help="Do not use GAT model to score docked poses.",
)

parser.add_argument(
    "--no-schnet",
    action="store_true",
    help="Do not use Schnet model to score docked poses.",
)

parser.add_argument(
    "--viz-target",
    type=str,
    choices=VizTargets.get_allowed_targets(),
    help="Target to write visualizations for, one of (sars2_mpro, mers_mpro, 7ene_mpro, 272_mpro, sars2_mac1)",
)

parser.add_argument(
    "--md", action="store_true", help="Whether to run MD after docking."
)

parser.add_argument(
    "--md-steps",
    action="store",
    type=int,
    default=2500000,
    help="Number of MD steps to run.",
)


def main():
    #########################
    # parse input arguments #
    #########################

    args = parser.parse_args()

    # if viz_target not specified, set to the target
    if not args.viz_target:
        args.viz_target = args.target

    # setup output directory
    output_dir = Path(args.output_dir)
    overwrote_dir = False
    if output_dir.exists():
        overwrote_dir = True
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logname = args.logname if args.logname else "single_target_docking"
    # setup logging
    logger_cls = FileLogger(logname, path=output_dir, stdout=True, level=logging.INFO)
    logger = logger_cls.getLogger()

    if overwrote_dir:
        logger.warning(f"Overwriting output directory: {output_dir}")

    logger.info(f"Start single target prep+docking at {datetime.now().isoformat()}")

    logger.info(f"IMPORTANT: Target: {args.target}")
    logger.info(f"Output directory: {output_dir}")

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

    #########################
    #    load input data    #
    #########################

    if args.postera:
        import os

        postera_api_key = os.getenv("POSTERA_API_KEY")

        if postera_api_key is None:
            raise ValueError("Environment variable POSTERA_API_KEY not found")

        logger.info("Postera API key found")
        logger.info(f"attempting to pull Molecule Set: {args.mols} from PostEra")

        from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI

        ms = MoleculeSetAPI("https://api.asap.postera.ai", "v1", postera_api_key)
        avail_molsets = ms.list_available()
        mols, molset_id = ms.get_molecules_from_id_or_name(args.mols)
        logger.info(
            f"Found {len(mols)} molecules in Molecule Set ID: {molset_id} with name: {avail_molsets[molset_id]}"
        )

        # make each molecule into a ExperimentalCompoundData object
        exp_data = [
            ExperimentalCompoundData(compound_id=mol.id, smiles=mol.smiles)
            for _, mol in mols.iterrows()
        ]
        used_3d = False

    else:
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

            exp_data = [
                ExperimentalCompoundData(compound_id=args.title, smiles=args.mols)
            ]
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
    if len(exp_data) == 0:
        logger.error("No molecules loaded.")
        raise ValueError("No molecules loaded.")

    if args.verbose:
        # we could make this just a debug statement but avoid looping over all molecules if not needed
        for exp in exp_data:
            logger.debug(f"Loaded molecule {exp.compound_id}: {exp.smiles}")

    #########################
    #     protein prep      #
    #########################

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

    # load the receptor

    receptor_name = receptor.stem
    logger.info(f"Loaded receptor {receptor_name} from {receptor}")

    xtal = CrystalCompoundData(
        dataset=Path(args.receptor).stem, str_fn=str(args.receptor)
    )

    prep_input_schema = PreppedTarget(
        source=xtal,
        output_name=str(receptor_name),
        active_site_chain=args.active_site_chain,
        oe_active_site_residue=args.oe_active_site_residue,
        molecule_filter=MoleculeFilter(
            components_to_keep=args.components_to_keep,
            ligand_chain=args.ligand_chain,
            protein_chains=args.protein_chains,
        ),
    )

    targets = PreppedTargets.from_list([prep_input_schema])
    targets.to_json(prep_dir / "input_targets.json")

    # setup prep options

    # check the reference structure exists
    if args.ref_prot is not None:
        if not Path(args.ref_prot).exists():
            raise ValueError(f"Reference protein file does not exist: {args.ref_prot}")
        else:
            logger.info(f"Using reference protein: {args.ref_prot}")

    reference_structure = args.ref_prot if args.ref_prot else receptor

    prep_opts = PrepOpts(
        ref_fn=reference_structure,
        ref_chain=args.ref_chain,
        loop_db=args.loop_db,
        seqres_yaml=args.seqres_yaml,
        output_dir=prep_dir,
        make_design_unit=True,
    )
    logger.info(f"Prep options: {prep_opts}")

    # add prep opts to the prepping function
    protein_prep_workflow_with_opts = partial(
        protein_prep_workflow, prep_opts=prep_opts
    )
    # there is only one target
    input_target = targets.iterable[0]

    # run prep
    prepped_targets = protein_prep_workflow_with_opts(input_target)
    prepped_targets = PreppedTargets.from_list([prepped_targets])
    prepped_targets.to_json(prep_dir / "output_targets.json")
    output_target = prepped_targets.iterable[0]

    if output_target.failed:
        raise ValueError("Protein prep failed.")
    output_target_du = output_target.design_unit
    if not output_target_du.exists():
        raise ValueError(f"Design unit does not exist: {output_target_du}")
    protein_path = output_target.protein
    if not protein_path.exists():
        raise ValueError(f"Protein file does not exist: {protein_path}")

    logger.info(f"Finished prepping receptor at {datetime.now().isoformat()}")

    # do dask here as it is the first bit of the workflow that is parallel
    if args.dask:
        logger.info("Using dask to parallelise docking")
        # set timeout to None so workers don't get killed on long timeouts
        from dask import config as cfg
        from dask.distributed import Client

        cfg.set({"distributed.scheduler.worker-ttl": None})

        if args.dask_lilac:
            from dask_jobqueue import LSFCluster

            logger.info("Using dask-jobqueue to run on lilac")
            logger.warning(
                "make sure you have a config file for lilac's dask-jobqueue cluster installed in your environment, contact @hmacdope"
            )

            logger.info("Finding dual IP interfaces")
            # find dual IP interfaces excluding lo loopback interface
            # technically dask only needs IPV4 but helps us find the right ones
            # easier. If you have a better way of doing this please let me know!
            exclude = ["lo"]
            interfaces = get_interfaces_with_dual_ip(exclude=exclude)
            logger.info(f"Found IP interfaces: {interfaces}")
            if len(interfaces) == 0:
                raise ValueError("Must have at least one network interface to run dask")
            if len(interfaces) > 1:
                logger.warning(
                    f"Found more than one IP interface: {interfaces}, using the first one"
                )
            interface = interfaces[0]
            logger.info(f"Using interface: {interface}")

            # NOTE you will need a config file that defines the dask-jobqueue for the cluster
            cluster = LSFCluster(
                interface=interface, scheduler_options={"interface": interface}
            )

            logger.info(f"dask config : {dask.config.config}")

            # assume we will have about 10 jobs, they will be killed if not used
            cluster.scale(10)
            # cluster is adaptive, and will scale between 5 and 40 workers depending on load
            # don't set it too low as then the cluster can scale down before needing to scale up again very rapidly
            # which can cause thrashing in the LSF queue
            cluster.adapt(minimum=5, maximum=40, interval="60s", target_duration="40s")
            client = Client(cluster)
        else:
            client = Client()
        logger.info("Dask client created ...")
        logger.info(client.dashboard_link)
        logger.info(
            "strongly recommend you open the dashboard link in a browser tab to monitor progress"
        )

    # setup docking
    logger.info(f"Starting docking setup at {datetime.now().isoformat()}")

    #########################
    #        docking        #
    #########################

    dock_dir = output_dir / "docking"
    dock_dir.mkdir(parents=True, exist_ok=True)
    intermediate_files.append(dock_dir)

    # ML stuff for docking, fill out others as we make them
    logger.info("Setup ML for docking")
    gat_model_string = "asapdiscovery-GAT-2023.05.09"
    schnet_model_string = "asapdiscovery-schnet-2023.04.29"

    if args.no_gat:
        gat_model = None
        logger.info("Not using GAT model")
    else:
        from asapdiscovery.ml.inference import GATInference  # noqa: E402

        gat_model = GATInference(gat_model_string)
        logger.info(f"Using GAT model: {gat_model_string}")

    if args.no_schnet:
        schnet_model = None
        logger.info("Not using Schnet model")
    else:
        from asapdiscovery.ml.inference import SchnetInference  # noqa: E402

        schnet_model = SchnetInference(schnet_model_string)
        logger.info(f"Using Schnet model: {schnet_model_string}")

    # use partial to bind the ML models to the docking function
    dock_and_score_pose_oe_ml = partial(
        dock_and_score_pose_oe,
        GAT_model=gat_model,
        schnet_model=schnet_model,
        allow_low_posit_prob=True,
        allow_final_clash=True,
    )

    if args.dask:
        dock_and_score_pose_oe_ml = dask.delayed(dock_and_score_pose_oe_ml)

    # run docking
    logger.info(f"Running docking at {datetime.now().isoformat()}")

    results = []

    # if we havn't already made the OE representation of the molecules, do it now
    if not used_3d:
        logger.info("Loading molecules from SMILES")
        oe_mols = exp_data_to_oe_mols(exp_data)
    else:
        logger.info("Using 3D molecules from input")

    if args.no_omega:
        omega = False
    else:
        omega = True

    for mol, compound in zip(oe_mols, exp_data):
        logger.debug(f"Running docking for {compound.compound_id}")
        if args.debug:
            # check smiles match
            if compound.smiles != oechem.OEMolToSmiles(mol):
                raise ValueError(
                    f"SMILES mismatch between {compound.compound_id} and {mol.GetTitle()}"
                )
        res = dock_and_score_pose_oe_ml(
            dock_dir / f"{compound.compound_id}_{receptor_name}",
            compound.compound_id,
            str(output_target_du),
            logname,
            f"{compound.compound_id}_{receptor_name}",
            load_openeye_design_unit(str(output_target_du)),
            mol,
            args.docking_sys.lower(),
            args.relax.lower(),
            args.posit_method.lower(),
            f"{compound.compound_id}_{receptor_name}",
            omega,
            args.num_poses,
        )
        results.append(res)

    if args.dask:  # make concrete
        results = dask.compute(*results)

    ###########################
    # wrangle docking results #
    ###########################

    logger.info(f"Finished docking at {datetime.now().isoformat()}")
    logger.info(f"Docking finished for {len(results)} runs.")

    # save results
    results_df, csv = make_docking_result_dataframe(results, output_dir, save_csv=True)

    logger.info(f"Saved results to {csv}")
    logger.info(f"Finish single target prep+docking at {datetime.now().isoformat()}")

    poses_dir = output_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting docking result processing at {datetime.now().isoformat()}")
    logger.info(f"Processing {len(results_df)} docking results")

    ###########################
    # pose HTML visualization #
    ###########################

    # only write out visualizations and do MD for the best posit score for each ligand
    # sort by posit score

    sorted_df = results_df.sort_values(by=["Docking_Confidence_POSIT"], ascending=False)
    top_posit = sorted_df.drop_duplicates(subset=["_ligand_id"], keep="first")
    # save with the failed ones in so its clear which ones failed if any did
    top_posit.to_csv(
        output_dir / f"poses_{args.target}_sorted_posit_prob.csv", index=False
    )
    n_total = len(top_posit)

    # IMPORTANT: only keep the ones that worked for the rest of workflow
    top_posit = top_posit[top_posit._docked_file != ""]
    if args.debug:
        # save debug csv if needed
        top_posit.to_csv(
            output_dir / f"poses_{args.target}_sorted_posit_prob_succeded.csv",
            index=False,
        )

    n_succeded = len(top_posit)
    n_failed = n_total - n_succeded

    logger.info(
        f"IMPORTANT: docking failed for {n_failed} poses / {n_total} total poses "
    )

    logger.info(
        f"Writing out visualization for top pose for each ligand (n={len(top_posit)})"
    )

    # add pose output column
    top_posit["_outpath_pose"] = top_posit["_ligand_id"].apply(
        lambda x: poses_dir / Path(x) / "visualization.html"
    )

    if args.dask:
        logger.info("Running HTML visualization with Dask")
        outpaths = []

        @dask.delayed
        def dask_html_adaptor(pose, outpath):
            html_visualiser = HTMLVisualizer(
                [pose],
                [outpath],
                args.viz_target,
                protein_path,
                logger=logger,
            )
            output_paths = html_visualiser.write_pose_visualizations()

            if len(output_paths) != 1:
                raise ValueError(
                    "Somehow got more than one output path from HTMLVisualizer"
                )
            return output_paths[0]

        for pose, output_path in zip(
            top_posit["_docked_file"], top_posit["_outpath_pose"]
        ):
            outpath = dask_html_adaptor(pose, output_path)
            outpaths.append(outpath)

        outpaths = client.compute(outpaths)
        outpaths = client.gather(outpaths)

    else:
        logger.info("Running HTML visualization in serial")
        html_visualiser = HTMLVisualizer(
            top_posit["_docked_file"],
            top_posit["_outpath_pose"],
            args.viz_target,
            protein_path,
            logger=logger,
        )
        html_visualiser.write_pose_visualizations()

    if args.dask:
        if args.dask_lilac:
            logger.info("Checking if we need to spawn a new client")
        else:  # cleanup CPU dask client and spawn new GPU-CUDA client
            client.close()

    ###########################
    #   pose MD simulation    #
    ###########################

    if args.md:
        logger.info(f"Running MD on top pose for each ligand (n={len(top_posit)})")

        if args.dask:
            if args.dask_lilac:
                logger.info(
                    "dask lilac setup means we don't need to spawn a new client"
                )
            else:
                logger.info("Starting seperate Dask GPU client")
                # spawn new GPU client
                from dask.distributed import Client
                from dask_cuda import LocalCUDACluster

                cluster = LocalCUDACluster()
                client = Client(cluster)
                logger.info(client.dashboard_link)
                logger.info(
                    "strongly recommend you open the dashboard link in a browser tab to monitor progress"
                )

        md_dir = output_dir / "md"
        md_dir.mkdir(parents=True, exist_ok=True)
        intermediate_files.append(md_dir)

        top_posit["_outpath_md"] = top_posit["_ligand_id"].apply(
            lambda x: md_dir / Path(x)
        )
        logger.info(f"Starting MD at {datetime.now().isoformat()}")

        reporting_interval = 1250

        if args.dask:
            logger.info("Running MD with Dask")

            # make a dask delayed function that runs the MD
            # must make the simulator inside the loop as it is not serialisable
            @dask.delayed
            def dask_md_adaptor(pose, protein_path, output_path):
                simulator = VanillaMDSimulator(
                    [pose],
                    protein_path,
                    logger=logger,
                    output_paths=[output_path],
                    num_steps=args.md_steps,
                    reporting_interval=reporting_interval,
                )
                retcode = simulator.run_all_simulations()
                if len(retcode) != 1:
                    raise ValueError(
                        "Somehow ran more than one simulation and got more than one retcode"
                    )
                return retcode[0]

            # make a list of simulator results that we then compute in parallel
            retcodes = []
            for pose, output_path in zip(
                top_posit["_docked_file"], top_posit["_outpath_md"]
            ):
                retcode = dask_md_adaptor(pose, protein_path, output_path)
                retcodes.append(retcode)

            # run in parallel sending out a bunch of Futures
            retcodes = client.compute(retcodes)
            # gather results back to the client, blocking until all are done
            retcodes = client.gather(retcodes)

        else:
            # don't do this if you can avoid it
            logger.info("Running MD with in serial")
            logger.warning("This will take a long time")
            simulator = VanillaMDSimulator(
                top_posit["_docked_file"],
                protein_path,
                logger=logger,
                output_paths=top_posit["_outpath_md"],
                num_steps=args.md_steps,
                reporting_interval=reporting_interval,
            )
            simulator.run_all_simulations()

        logger.info(f"Finished MD at {datetime.now().isoformat()}")

        ###########################
        #   MD GIF visualization  #
        ###########################

        logger.info("making GIF visualizsations")

        gif_dir = output_dir / "gif"
        gif_dir.mkdir(parents=True, exist_ok=True)

        top_posit["_outpath_md_sys"] = top_posit["_outpath_md"].apply(
            lambda x: Path(x) / "minimized.pdb"
        )

        top_posit["_outpath_md_traj"] = top_posit["_outpath_md"].apply(
            lambda x: Path(x) / "traj.xtc"
        )

        top_posit["_outpath_gif"] = top_posit["_ligand_id"].apply(
            lambda x: gif_dir / Path(x) / "trajectory.gif"
        )
        # take only last .5ns of trajectory to get nicely equilibrated pose.

        n_snapshots = int(args.md_steps / reporting_interval)

        # take last 100 snapshots
        if n_snapshots < 100:
            start = 1
        else:
            start = n_snapshots - 99

        @dask.delayed
        def dask_gif_adaptor(traj, system, outpath):
            gif_visualizer = GIFVisualizer(
                [traj],
                [system],
                [outpath],
                args.viz_target,
                frames_per_ns=200,
                smooth=5,
                start=start,
                logger=logger,
            )
            output_paths = gif_visualizer.write_traj_visualizations()

            if len(output_paths) != 1:
                raise ValueError(
                    "Somehow got more than one output path from GIFVisualizer"
                )
            return output_paths[0]

        if args.dask:
            logger.info("Running GIF visualization with Dask")
            outpaths = []
            for traj, system, outpath in zip(
                top_posit["_outpath_md_traj"],
                top_posit["_outpath_md_sys"],
                top_posit["_outpath_gif"],
            ):
                outpath = dask_gif_adaptor(traj, system, outpath)
                outpaths.append(outpath)

            # run in parallel sending out a bunch of Futures
            outpaths = client.compute(outpaths)
            # gather results back to the client, blocking until all are done
            outpaths = client.gather(outpaths)

        else:
            logger.info("Running GIF visualization in serial")
            logger.warning("This will take a long time")
            gif_visualiser = GIFVisualizer(
                top_posit["_outpath_md_traj"],
                top_posit["_outpath_md_sys"],
                top_posit["_outpath_gif"],
                args.viz_target,
                frames_per_ns=200,
                smooth=5,
                start=start,
                logger=logger,
            )
            gif_visualiser.write_traj_visualizations()

    renamed_top_posit = drop_and_rename_docking_output_cols_for_manifold(
        top_posit, args.target, manifold_validate=True, allow=["ligand_id"]
    )
    # save to final CSV renamed for target
    renamed_top_posit.to_csv(output_dir / f"poses_{args.target}_final.csv", index=False)

    if args.postera_upload:
        if not args.postera:
            raise ValueError("Must use --postera to upload to PostEra")
        logger.info("Uploading results to PostEra")

        ms.update_molecules_from_df_with_manifold_validation(
            molecule_set_id=molset_id,
            df=renamed_top_posit,
            id_field="_ligand_id",
            smiles_field="SMILES",
            overwrite=True,
            debug_df_path=output_dir / "postera_uploaded.csv",
        )
        logger.info("Finished uploading results to PostEra")

    if args.cleanup:
        if len(intermediate_files) > 0:
            logger.warning("Removing intermediate files.")
            for path in intermediate_files:
                shutil.rmtree(path)
    else:
        logger.info("Keeping intermediate files.")


if __name__ == "__main__":
    main()
