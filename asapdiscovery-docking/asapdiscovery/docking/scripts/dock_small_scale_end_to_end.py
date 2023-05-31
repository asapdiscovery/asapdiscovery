import argparse
import logging
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path  # noqa: F401
from typing import List  # noqa: F401


import dask
import yaml
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    oechem,
    save_openeye_pdb,
    save_openeye_sdf,
    split_openeye_design_unit,
)
from asapdiscovery.data.schema import CrystalCompoundData, ExperimentalCompoundData
from asapdiscovery.data.utils import (
    exp_data_to_oe_mols,
    is_valid_smiles,
    oe_load_exp_from_file,
)
from asapdiscovery.data.execution_utils import get_interfaces_with_dual_ip
from asapdiscovery.dataviz.gif_vis import GIFVisualiser
from asapdiscovery.dataviz.html_vis import HTMLVisualiser
from asapdiscovery.docking import make_docking_result_dataframe
from asapdiscovery.docking import prep_mp as oe_prep_function
from asapdiscovery.docking.mcs import rank_structures_openeye  # noqa: F401
from asapdiscovery.docking.mcs import rank_structures_rdkit  # noqa: F401
from asapdiscovery.docking.scripts.run_docking_oe import mp_func as oe_docking_function
from asapdiscovery.simulation.simulate import VanillaMDSimulator

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
    "--dask",
    action="store_true",
    help=("use dask to parallelise docking"),
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

parser.add_argument(
    "--md-steps",
    action="store",
    type=int,
    default=2500000,
    help="Number of MD steps to run.",
)


def main():
    args = parser.parse_args()

    # setup output directory
    output_dir = Path(args.output_dir)
    overwrote_dir = False
    if output_dir.exists():
        overwrote_dir = True
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logname = args.logname if args.logname else "single_target_docking"
    # setup logging
    logger_cls = FileLogger(logname, path=output_dir, stdout=True)
    logger = logger_cls.getLogger()

    if overwrote_dir:
        logger.warning(f"Overwriting output directory: {output_dir}")

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

    if args.dask:
        logger.info("Using dask to parallelise docking")
        from dask.distributed import Client

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
            logger.info(f"dask config : {dask.config.config}")

            # NOTE you will need a config file that defines the dask-jobqueue for the cluster
            cluster = LSFCluster(
                interface=interface, scheduler_options={"interface": interface}
            )

            # assume we will have about 10 jobs, they will be killed if not used
            cluster.scale(10)
            # cluster is adaptive, and will scale between 5 and 40 workers depending on load
            # don't set it too low as then the cluster can scale down before needing to scale up again very rapidly
            # which can cause thrashing in the LSF queue
            cluster.adapt(minimum=5, maximum=40, interval="10s", target_duration="60s")
            client = Client(cluster)
        else:
            client = Client()
        logger.info("Dask client created ...")
        logger.info(client.dashboard_link)
        logger.info(
            "strongly recommend you open the dashboard link in a browser tab to monitor progress"
        )

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
    if len(exp_data) == 0:
        logger.error("No molecules loaded.")
        raise ValueError("No molecules loaded.")

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

    # extract the ligand, protein and complex
    lig, protein, lig_prot_complex = split_openeye_design_unit(du)

    # write out the protein
    protein_path = prep_dir / f"{receptor_name}" / f"{receptor_name}_protein.pdb"
    logger.info(f"Writing out protein to {protein_path}")
    save_openeye_pdb(protein, str(protein_path))

    # write out the complex
    complex_path = prep_dir / f"{receptor_name}" / f"{receptor_name}_complex.pdb"
    logger.info(f"Writing out complex to {complex_path}")
    save_openeye_pdb(lig_prot_complex, str(complex_path))

    # write out the ligand
    ligand_path = prep_dir / f"{receptor_name}" / f"{receptor_name}_ligand.sdf"
    logger.info(f"Writing out ligand to {ligand_path}")
    save_openeye_sdf(lig, str(ligand_path))

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

    if args.dask:
        full_oe_docking_function = dask.delayed(full_oe_docking_function)

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
        res = full_oe_docking_function(
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
        results.append(res)

    if args.dask:  # make concrete
        results = dask.compute(*results)

    logger.info(f"Finished docking at {datetime.now().isoformat()}")
    logger.info(f"Docking finished for {len(results)} runs.")

    # save results
    results_df, csv = make_docking_result_dataframe(results, output_dir, save_csv=True)

    logger.info(f"Saved results to {csv}")
    logger.info(f"Finish single target prep+docking at {datetime.now().isoformat()}")

    poses_dir = output_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    intermediate_files.append(poses_dir)
    logger.info(f"Starting docking result processing at {datetime.now().isoformat()}")
    logger.info(f"Processing {len(results_df)} docking results")

    # only write out visualisation for the best posit score for each ligand
    # sort by posit  score
    sorted_df = results_df.sort_values(by=["POSIT_prob"], ascending=False)
    top_posit = sorted_df.drop_duplicates(subset=["ligand_id"], keep="first")
    # save with the failed ones in so its clear which ones failed
    top_posit.to_csv(output_dir / "top_poses.csv", index=False)
    # only keep the ones that worked for the rest of workflow
    top_posit = top_posit[top_posit.docked_file != ""]
    top_posit.to_csv(output_dir / "top_poses_clean.csv", index=False)

    logger.info(
        f"Writing out visualisation for top pose for each ligand (n={len(top_posit)})"
    )

    # add pose output column
    top_posit["outpath_pose"] = top_posit["ligand_id"].apply(
        lambda x: poses_dir / Path(x) / "visualisation.html"
    )

    html_visualiser = HTMLVisualiser(
        top_posit["docked_file"],
        top_posit["outpath_pose"],
        args.target,
        protein_path,
        logger=logger,
    )
    html_visualiser.write_pose_visualisations()

    del html_visualiser

    if args.dask:
        if args.dask_lilac:
            logger.info("Checking if we need to spawn a new client")
        else:  # cleanup CPU dask client and spawn new GPU-CUDA client
            client.close()

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

        top_posit["outpath_md"] = top_posit["ligand_id"].apply(
            lambda x: md_dir / Path(x)
        )
        logger.info(f"Starting MD at {datetime.now().isoformat()}")

        reporting_interval = 1250

        if args.dask:
            logger.info("Running MD with Dask")

            # make a dask delayed function that runs the MD
            # must make the simulator inside the loop as it is not serialisable
            @dask.delayed
            def dask_md_adaptor(pose, protein_path, logger, output_path):
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
                top_posit["docked_file"], top_posit["outpath_md"]
            ):
                retcode = dask_md_adaptor(pose, protein_path, logger, output_path)
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
                top_posit["docked_file"],
                protein_path,
                logger=None,
                output_paths=top_posit["outpath_md"],
                num_steps=args.md_steps,
                reporting_interval=reporting_interval,
            )
            simulator.run_all_simulations()

        logger.info(f"Finished MD at {datetime.now().isoformat()}")

        logger.info("making GIF visualisations")

        gif_dir = output_dir / "gif"
        gif_dir.mkdir(parents=True, exist_ok=True)
        intermediate_files.append(gif_dir)

        top_posit["outpath_md_sys"] = top_posit["outpath_md"].apply(
            lambda x: Path(x) / "minimized.pdb"
        )

        top_posit["outpath_md_traj"] = top_posit["outpath_md"].apply(
            lambda x: Path(x) / "traj.xtc"
        )

        top_posit["outpath_gif"] = top_posit["ligand_id"].apply(
            lambda x: gif_dir / Path(x) / "trajectory.gif"
        )
        # take only last .5ns of trajectory to get nicely equilibrated pose.

        n_snapshots = int(args.md_steps / reporting_interval)

        # take last 100 snapshots
        if n_snapshots < 100:
            start = 1
        else:
            start = n_snapshots - 100

        @dask.delayed
        def dask_gif_adaptor(traj, system, outpath):

            gif_visualiser = GIFVisualiser(
                [traj],
                [system],
                [outpath],
                args.target,
                smooth=5,
                start=start,
                logger=logger,
            )
            output_paths = gif_visualiser.write_traj_visualisations()

            if len(output_paths) != 1:
                raise ValueError(
                    "Somehow got more than one output path from GIFVisualiser"
                )
            return output_paths[0]

        if args.dask:
            logger.info("Running GIF visualisation with Dask")
            outpaths = []
            for traj, system, outpath in zip(
                top_posit["outpath_md_traj"],
                top_posit["outpath_md_sys"],
                top_posit["outpath_gif"],
            ):
                outpath = dask_gif_adaptor(traj, system, outpath)
                outpaths.append(outpath)

            # run in parallel sending out a bunch of Futures
            outpaths = client.compute(outpaths)
            # gather results back to the client, blocking until all are done
            outpaths = client.gather(outpaths)

        else:
            logger.info("Running GIF visualisation in serial")
            logger.warning("This will take a long time")
            gif_visualiser = GIFVisualiser(
                top_posit["outpath_md_traj"],
                top_posit["outpath_md_sys"],
                top_posit["outpath_gif"],
                args.target,
                smooth=5,
                start=start,
                logger=logger,
            )
            gif_visualiser.write_traj_visualisations()

    if args.cleanup:
        if len(intermediate_files) > 0:
            logger.warning("Removing intermediate files.")
            for path in intermediate_files:
                shutil.rmtree(path)
    else:
        logger.info("Keeping intermediate files.")


if __name__ == "__main__":
    main()
