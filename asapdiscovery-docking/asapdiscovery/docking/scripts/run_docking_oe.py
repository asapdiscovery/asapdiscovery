"""
Script to dock an SDF file of ligands to prepared structures. Example usage:
python run_docking_oe.py \
-e /path/to/experimental_data.json \
-r /path/to/receptors/*.oedu \
-s /path/to/mcs_sort_results.pkl \
-o /path/to/docking/output/

Example usage with a custom regex for parsing your DU filenames:
Suppose your receptors are named
 - /path/to/receptors/Structure0_0.oedu
 - /path/to/receptors/Structure0_1.oedu
 - /path/to/receptors/Structure0_2.oedu
 - ...
where each Structure<i> is a unique crystal structure, and each Structure<i>_<j> is a
different DesignUnit for that structure. You might construct your regex as
'(Compound[0-9]+)_[0-9]+', which will capture the Structure<i> as the unique structure
ID, and Structure<i>_<j> as the full name. Note that single quotes should be used around
the regex in order to avoid any accidental wildcard expansion by the OS:
python run_docking_oe.py \
-e /path/to/experimental_data.json \
-r /path/to/receptors/*.oedu \
-s /path/to/mcs_sort_results.pkl \
-o /path/to/docking/output/ \
-re '(Compound[0-9]+)_[0-9]+'
"""
import argparse
import logging
import multiprocessing as mp
import os
import pickle as pkl
from concurrent.futures import TimeoutError
from datetime import datetime
from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import pandas
import pebble
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (  # noqa: E402
    combine_protein_ligand,
    load_openeye_sdf,
    oechem,
    save_openeye_pdb,
    save_openeye_sdf,
)
from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate  # noqa: E402
from asapdiscovery.data.utils import check_filelist_has_elements  # noqa: E402
from asapdiscovery.docking.docking import POSIT_METHODS, run_docking_oe  # noqa: E402
from asapdiscovery.modeling.modeling import split_openeye_design_unit


def check_results(d):
    """
    Check if results exist already so we can skip.

    Parameters
    ----------
    d : str
        Directory

    Returns
    -------
    bool
        Results already exist
    """
    if (not os.path.isfile(os.path.join(d, "docked.sdf"))) or (
        not os.path.isfile(os.path.join(d, "results.pkl"))
    ):
        return False

    try:
        _ = load_openeye_sdf(os.path.join(d, "docked.sdf"))
    except Exception:
        return False

    try:
        _ = pkl.load(open(os.path.join(d, "results.pkl"), "rb"))
    except Exception:
        return False

    return True


def load_dus(fn_dict, log_name):
    """
    Load all present oedu files.

    Parameters
    ----------
    fn_dict : Dict[str, str]
        Dictionary mapping full DesignUnit name (with chain) to full filename

    Returns
    -------
    Dict[str, oechem.OEDesignUnit]
        Dictionary mapping full Mpro name/compound id (including chain) to its
        design unit
    """
    logger = logging.getLogger(log_name)

    du_dict = {}
    for full_name, fn in fn_dict.items():
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(fn, du):
            logger.error(f"Failed to read DesignUnit {fn}")
            continue
        du_dict[full_name] = du

    logger.info(f"{len(du_dict.keys())} design units loaded")
    return du_dict


def mp_func(
    out_dir,
    lig_name,
    du_name,
    log_name,
    compound_name,
    du,
    *args,
    GAT_model=None,
    schnet_model=None,
    **kwargs,
):
    """
    Wrapper function for multiprocessing. Everything other than the named args
    will be passed directly to run_docking_oe.

    Parameters
    ----------
    out_dir : str
        Output file
    lig_name : str
        Ligand name
    du_name : str
        DesignUnit name
    log_name : str
        High-level logger name
    compound_name : str
        Compound name, used for error messages if given
    GAT_model : GATInference, optional
        GAT model to use for inference. If None, will not perform inference.
    schnet_model : SchNetInference, optional
        SchNet model to use for inference. If None, will not perform inference.

    Returns
    -------
    """
    logname = f"{log_name}.{compound_name}"

    before = datetime.now().isoformat()
    if check_results(out_dir):
        logger = FileLogger(logname, path=str(out_dir)).getLogger()
        logger.info(f"Found results for {compound_name}")
        after = datetime.now().isoformat()
        results = pkl.load(open(os.path.join(out_dir, "results.pkl"), "rb"))
        logger.info(f"Start: {before}, End: {after}")
        return results
    else:
        os.makedirs(out_dir, exist_ok=True)
        logger = FileLogger(logname, path=str(out_dir)).getLogger()
        logger.info(f"No results for {compound_name} found, running docking")
        errfs = oechem.oeofstream(os.path.join(out_dir, f"openeye_{logname}-log.txt"))
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)
        oechem.OEThrow.Info(f"Starting docking for {logname}")

    success, posed_mol, docking_id = run_docking_oe(
        du, *args, log_name=log_name, **kwargs
    )
    if success:
        out_fn = os.path.join(out_dir, "docked.sdf")
        save_openeye_sdf(posed_mol, out_fn)

        rmsds = []
        posit_probs = []
        posit_methods = []
        chemgauss_scores = []
        schnet_scores = []

        # grab the du passed in and split it
        lig, prot, complex = split_openeye_design_unit(du.CreateCopy())

        for conf in posed_mol.GetConfs():
            rmsds.append(float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_RMSD")))
            posit_probs.append(
                float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_POSIT"))
            )
            posit_methods.append(
                oechem.OEGetSDData(conf, f"Docking_{docking_id}_POSIT_method")
            )
            chemgauss_scores.append(
                float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_Chemgauss4"))
            )
            if schnet_model is not None:
                # TODO: this is a hack, we should be able to do this without saving
                # the file to disk see # 253
                outpath = Path(out_dir) / Path(".posed_mol_schnet_temp.pdb")
                # join with the protein only structure
                combined = combine_protein_ligand(prot, conf)
                pdb_temp = save_openeye_pdb(combined, outpath)
                schnet_score = schnet_model.predict_from_structure_file(pdb_temp)
                schnet_scores.append(schnet_score)
                outpath.unlink()
            else:
                schnet_scores.append(np.nan)

        smiles = oechem.OEGetSDData(conf, "SMILES")
        clash = int(oechem.OEGetSDData(conf, f"Docking_{docking_id}_clash"))
        if GAT_model is not None:
            GAT_score = GAT_model.predict_from_smiles(smiles)
        else:
            GAT_score = np.nan

    else:
        out_fn = ""
        rmsds = [-1.0]
        posit_probs = [-1.0]
        posit_methods = [""]
        chemgauss_scores = [-1.0]
        clash = -1
        smiles = "None"
        GAT_score = np.nan
        schnet_scores = [np.nan]

    results = [
        (
            lig_name,
            du_name,
            out_fn,
            i,
            rmsd,
            prob,
            method,
            chemgauss,
            clash,
            smiles,
            GAT_score,
            schnet,
        )
        for i, (rmsd, prob, method, chemgauss, schnet) in enumerate(
            zip(rmsds, posit_probs, posit_methods, chemgauss_scores, schnet_scores)
        )
    ]

    pkl.dump(results, open(os.path.join(out_dir, "results.pkl"), "wb"))
    after = datetime.now().isoformat()
    logger.info(f"Start: {before}, End: {after}")
    return results


def parse_du_filenames(receptors, regex, log_name, basefile="predocked.oedu"):
    """
    Parse list of DesignUnit filenames and extract identifiers using the given regex.
    `regex` should have one capturing group (which can be the entire string if desired).

    Parameters
    ----------
    receptors : Union[List[str], str]
        Either list of DesignUnit filenames, or glob/directory/file to load from. If a
        file is passed, will assume this is a CSV file and will load from the
        "Docked_File" column using `basefile`
    regex : str
        Regex string for parsing
    basefile : str, default="predocked.oedu"
        If a CSV file is passed for `receptors`, this is the base filename that will be
        appended to every directory found in the "Docked_File" column

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping Mpro dataset name/compound id to list of full
        Mpro names/compound ids (with chain)
    Dict[str, str]
        Dictionary mapping full name (with chain) to full filename
    """
    from asapdiscovery.data.utils import construct_regex_function

    logger = logging.getLogger(log_name)

    # First get full list of filenames
    if type(receptors) is list:
        logger.info("Using files as given")
        all_fns = receptors
    elif os.path.isdir(receptors):
        logger.info(f"Using {receptors} as directory")
        all_fns = [
            os.path.join(receptors, fn)
            for _, _, files in os.walk(receptors)
            for fn in files
            if fn[-4:] == "oedu"
        ]

    elif os.path.isfile(receptors):
        logger.info(f"Using {receptors} as individual file")
        file_extn = os.path.splitext(receptors)[1]
        if file_extn == ".csv":
            logger.info(f"Using {receptors} as CSV file")
            df = pandas.read_csv(receptors)
            try:
                all_fns = [
                    os.path.join(os.path.dirname(fn), basefile)
                    for fn in df["Docked_File"]
                ]
            except KeyError:
                raise ValueError("Docked_File column not found in given CSV file.")
        elif file_extn == ".oedu":
            logger.info(f"Using {receptors} as single DesignUnit file")
            all_fns = [receptors]
        else:
            raise ValueError("File must be either .csv or .oedu")
    else:
        logger.info(f"Using {receptors} as glob")
        all_fns = glob(receptors)
        logger.info(all_fns)

    # check that we actually have loaded in prepped receptors.
    check_filelist_has_elements(all_fns, tag="prepped receptors")
    logger.info(f"{len(all_fns)} DesignUnit files found")

    # Build regex search function
    regex_func = construct_regex_function(regex, ret_groups=True)
    # Perform searches and build dicts
    dataset_dict = {}
    fn_dict = {}
    for fn in all_fns:
        try:
            full_name, dataset = regex_func(fn)
        except ValueError:
            logger.error(f"No regex match found for {fn}")
            continue

        try:
            dataset = dataset[0]
        except IndexError:
            raise ValueError(f"No capturing group in regex {regex}")

        try:
            dataset_dict[dataset].append(full_name)
        except KeyError:
            dataset_dict[dataset] = [full_name]
        fn_dict[full_name] = fn

    return dataset_dict, fn_dict


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument("-l", "--lig_file", help="SDF file containing ligands.")
    parser.add_argument(
        "-e",
        "--exp_file",
        help="JSON file containing ExperimentalCompoundDataUpdate object.",
    )
    parser.add_argument(
        "-r",
        "--receptor",
        required=True,
        nargs="+",
        help=(
            "Path/glob to prepped receptor(s), or CSV file containing receptor paths"
        ),
    )
    parser.add_argument(
        "-s",
        "--sort_res",
        help="Pickle file giving compound_ids, xtal_ids, and sort_idxs.",
    )
    parser.add_argument(
        "-re",
        "--regex",
        help=(
            "Regex for extracting DesignUnit identifiers from the "
            "OpenEye DesignUnit filenames."
        ),
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    # Performance arguments
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help=(
            "Number of concurrent processes to run. "
            "Set to <= 1 to disable multiprocessing."
        ),
    )
    parser.add_argument(
        "-m",
        "--timeout",
        type=int,
        default=300,
        help=(
            "Timeout (in seconds) for each docking thread. "
            "Setting to a number <=0 disables this feature."
        ),
    )
    parser.add_argument(
        "-t",
        "--top_n",
        type=int,
        default=1,
        help="Number of top matches to dock. Set to -1 to dock all.",
    )
    parser.add_argument(
        "-d",
        "--docking_sys",
        default="posit",
        help="Which docking system to use [posit, hybrid]. Defaults to posit.",
    )
    parser.add_argument(
        "-x",
        "--relax",
        default="none",
        help="When to run relaxation [none, clash, all]. Defaults to none.",
    )
    parser.add_argument(
        "-y",
        "--posit_method",
        type=str,
        default="all",
        choices=POSIT_METHODS,
        help="Whether to only use hybrid docking protocol in POSIT.",
    )
    parser.add_argument(
        "-c",
        "--by_compound",
        action="store_true",
        help="Load/store DesignUnits by compound_id instead of by Mpro dataset.",
    )
    parser.add_argument(
        "-g",
        "--omega",
        action="store_true",
        help="Use Omega conformer enumeration.",
    )
    parser.add_argument(
        "-p",
        "--num_poses",
        type=int,
        default=1,
        help="Number of poses to return from docking.",
    )
    parser.add_argument(
        "--debug_num",
        type=int,
        default=None,
        help="Number of docking runs to run. Useful for debugging and testing.",
    )

    parser.add_argument(
        "--max_failures",
        type=int,
        default=20,
        help="Maximum number of failed docking runs to allow before exiting.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out verbose logging.",
    )
    parser.add_argument(
        "-gat",
        "--gat",
        action="store_true",
        help="Whether to use GAT model to score docked poses.",
    )
    parser.add_argument(
        "-schnet",
        "--schnet",
        action="store_true",
        help="Whether to use Schnet model to score docked poses.",
    )
    parser.add_argument(
        "-log",
        "--log_name",
        type=str,
        default="run_docking_oe",
        help="Base name for high-level log file. Defaults to run_docking_oe, "
        "which enables propagation of log messages to the root logger.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    log_name = args.log_name

    # Parse symlinks in output_dir
    output_dir = Path(args.output_dir)

    # check that output_dir exists, otherwise create it
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = FileLogger("run_docking_oe", path=str(output_dir)).getLogger()
    logger.info(f"Output directory: {output_dir}")
    start = datetime.now().isoformat()
    logger.info(f"Starting run_docking_oe at {start}")

    if args.exp_file:
        logger.info("Loading experimental compounds from JSON file")
        import json

        # Load compounds
        exp_compounds = [
            c
            for c in ExperimentalCompoundDataUpdate(
                **json.load(open(args.exp_file))
            ).compounds
            if c.smiles is not None
        ]
        # Make OEGraphMol for each compound
        mols = []
        for c in exp_compounds:
            new_mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(new_mol, c.smiles)
            mols.append(new_mol)
    if args.lig_file:
        logger.info(f"Loading ligands from {args.lig_file}")
        if args.exp_file:
            logger.info(
                (
                    "WARNING: Arguments passed for both --exp_file and "
                    "--lig_file, using --exp_file."
                ),
            )
        else:
            # Load all ligands to dock
            ifs = oechem.oemolistream()
            ifs.open(args.lig_file)
            mols = [mol.CreateCopy() for mol in ifs.GetOEGraphMols()]
            ifs.close()
    elif args.exp_file is None:
        raise ValueError("Need to specify exactly one of --exp_file or --lig_file.")

    n_mols = len(mols)
    logger.info(f"Loaded {n_mols} ligands, proceeding with docking setup")

    # Set up ML model
    gat_model_string = "asapdiscovery-GAT-2023.05.09"
    if args.gat:
        from asapdiscovery.ml.inference import GATInference  # noqa: E402

        GAT_model = GATInference(gat_model_string)
        logger.info(f"Using GAT model: {gat_model_string}")
    else:
        logger.info("Skipping GAT model scoring")
        GAT_model = None  # noqa: F841

    schnet_model_string = "asapdiscovery-schnet-2023.04.29"
    if args.schnet:
        from asapdiscovery.ml.inference import SchnetInference  # noqa: E402

        schnet_model = SchnetInference(schnet_model_string)
        logger.info(f"Using Schnet model: {schnet_model_string}")
    else:
        logger.info("Skipping Schnet model scoring")
        schnet_model = None  # noqa: F841

    # The receptor args are captured as a list, but we still want to handle the case of
    #  a glob/directory/filename being passed. If there's only one thing in the list,
    #  assume it is a glob/directory/filename, and pull it out of the list so it's
    #  properly handled in `parse_du_filenames`
    if len(args.receptor) == 1:
        logger.info("Receptor argument is a glob/directory/filename")
        args.receptor = args.receptor[0]
    # Handle default regex
    if args.regex is None:
        logger.info("No regex specified, using default regex")
        if args.by_compound:
            from asapdiscovery.data.utils import MOONSHOT_CDD_ID_REGEX_CAPT

            args.regex = MOONSHOT_CDD_ID_REGEX_CAPT
            logger.info(
                f"--by_compound specified, using MOONSHOT_CDD_ID_REGEX_CAPT regex: {MOONSHOT_CDD_ID_REGEX_CAPT}"
            )

        else:
            from asapdiscovery.data.utils import MPRO_ID_REGEX_CAPT

            args.regex = MPRO_ID_REGEX_CAPT
            logger.info(f"Using MPRO_ID_REGEX_CAPT regex: {MPRO_ID_REGEX_CAPT}")
    else:
        logger.info(f"Using custom regex: {args.regex}")

    logger.info(
        f"Parsing receptor design units with arguments: {args.receptor}, {args.regex}"
    )
    dataset_dict, fn_dict = parse_du_filenames(args.receptor, args.regex, log_name)

    # Load all receptor DesignUnits
    logger.info("Loading receptor DesignUnits")
    du_dict = load_dus(fn_dict, log_name)
    logger.info(f"{n_mols} molecules found")
    logger.info(f"{len(du_dict.keys())} receptor structures found")

    if not n_mols > 0:
        raise ValueError("No ligands found")
    if not len(du_dict.keys()) > 0:
        raise ValueError("No receptor structures found")

    # Load sort indices if given
    if args.sort_res:
        logger.info(f"Loading sort results from {args.sort_res}")
        compound_ids, xtal_ids, sort_idxs = pkl.load(open(args.sort_res, "rb"))
        # If we're docking to all DUs, set top_n appropriately
        if args.top_n == -1:
            logging.info("Docking to all")
            args.top_n = len(xtal_ids)
        else:
            logging.info(f"Docking to top {args.top_n}")

        # Make sure that compound_ids match with experimental data if that's
        #  what we're using
        if args.exp_file:
            logger.info("Checking that sort results match experimental data")
            if not all(
                [
                    compound_id == c.compound_id
                    for (compound_id, c) in zip(compound_ids, exp_compounds)
                ]
            ):
                raise ValueError(
                    "Compound IDs in sort results do not match experimental data"
                )
            logger.info("Sort results match experimental data")
    else:
        logger.info("No sort results given")
        # Check to see if the SDF files have a Compound_ID Column
        if all(len(oechem.OEGetSDData(mol, "Compound_ID")) > 0 for mol in mols):
            logger.info("Using Compound_ID column from sdf file")
            compound_ids = [oechem.OEGetSDData(mol, "Compound_ID") for mol in mols]
        else:
            # Use index as compound_id
            logger.info("Using index as compound_id")
            compound_ids = [str(i) for i in range(n_mols)]
        # Get dataset values from DesignUnit filenames
        xtal_ids = list(dataset_dict.keys())
        # Arbitrary sort index, same for each ligand
        sort_idxs = [list(range(len(xtal_ids)))] * n_mols
        args.top_n = len(xtal_ids)

    # make multiprocessing args
    logger.info("Making multiprocessing args")
    mp_args = []

    # if we are failing to read all the design units lets capture that before we get too far
    failures = 0

    # figure out what we need to be skipping
    xtal_set = set(xtal_ids)
    xtal_set_str = "\n".join(list(xtal_set))
    logger.info(f"Set of xtal ids read from sorting or inferred:\n{xtal_set_str}")

    dataset_set = set(dataset_dict.keys())
    dataset_set_str = "\n".join(list(dataset_set))
    logger.info(f"Set of xtal ids read from receptor files:\n{dataset_set_str}")

    diff = xtal_set - dataset_set
    diff_str = "\n".join(list(diff))
    if len(diff) > 0:
        logger.warning(
            f"Xtals that are in sort indices but don't have matching receptors read from file:\n{diff_str}"
        )
        logger.warning(
            f"THESE XTALS in WILL BE SKIPPED likely due to missing receptor files.\nTHIS MAY BE NORMAL IF BREAKING A LARGE JOB INTO CHUNKS.\n{diff_str}"
        )

    skipped = []
    for i, m in enumerate(mols):
        dock_dus = []
        xtals = []
        for xtal in sort_idxs[i][: args.top_n]:
            if xtal_ids[xtal] not in dataset_dict:
                if args.verbose:
                    skipped.append(
                        f"Crystal: {xtal_ids[xtal]} Molecule_title: {m.GetTitle()}, Compound_ID: {compound_ids[i]}, Smiles: {oechem.OECreateIsoSmiString(m)}"
                    )
                failures += 1
                continue

            # Get the DU for each full Mpro name associated with this dataset
            dock_dus.extend([du_dict[x] for x in dataset_dict[xtal_ids[xtal]]])
            xtals.extend(dataset_dict[xtal_ids[xtal]])
        new_args = [
            (
                output_dir / f"{compound_ids[i]}_{x}",
                compound_ids[i],
                x,
                log_name,
                f"{compound_ids[i]}_{x}",
                du,
                m,
                args.docking_sys.lower(),
                args.relax.lower(),
                args.posit_method.lower(),
                f"{compound_ids[i]}_{x}",
                args.omega,
                args.num_poses,
            )
            for du, x in zip(dock_dus, xtals)
        ]
        mp_args.extend(new_args)

    if args.verbose:
        if len(skipped) > 0:
            logger.warning(f"Skipped {len(skipped)} receptor/ligand pairs")
            for s in skipped:
                logger.warning("Skipped pair: " + s)

    if len(mp_args) == 0:
        raise ValueError(
            "No MP args built, likely due to no xtals found, check logs and increase verbosity with --verbose for more info. "
        )

    if failures > 0:
        logger.info(
            f"MP args built, {len(mp_args)} total with {failures} failures, most likely due to skipped xtals.\n"
            "Use --verbose flag to find out more"
        )
    else:
        logger.info(f"{len(mp_args)} multiprocessing args built successfully.")

    if (args.debug_num is not None) and (args.debug_num > 0):
        logger.info(f"DEBUG MODE: Only running {args.debug_num} docking runs")
        mp_args = mp_args[: args.debug_num]

    # Apply ML arguments as kwargs to mp_func
    mp_func_ml_applied = partial(
        mp_func, GAT_model=GAT_model, schnet_model=schnet_model
    )

    if args.num_cores > 1:
        logger.info("Running docking using multiprocessing")
        # reset failures
        logging.info(f"max_failures for running docking using MP : {args.max_failures}")

        nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
        logger.info(f"CPUs: {mp.cpu_count()}")
        logger.info(f"N Processes: {len(mp_args)}")
        logger.info(f"N Cores: {args.num_cores}")
        logger.info(f"Running {len(mp_args)} docking runs over {nprocs} cores.")
        with pebble.ProcessPool(max_workers=nprocs) as pool:
            if args.timeout <= 0:
                args.timeout = None
            # Need to flip args structure for pebble
            res = pool.map(mp_func_ml_applied, *zip(*mp_args), timeout=args.timeout)

            # List to keep track of successful results
            results_list = []
            # List to keep track of which runs failed
            failed_runs = []

            # TimeoutError is only raised when we try to access the result. Do things
            #  this way so we can keep track of which compound:xtals timed out
            res_iter = res.result()
            for args_list in mp_args:
                docking_run_name = args_list[9]
                try:
                    cur_res = next(res_iter)
                    results_list += [cur_res]
                except StopIteration:
                    # We've reached the end of the results iterator so just break
                    break
                except TimeoutError:
                    # This compound:xtal combination timed out
                    logger.error("Docking timed out for", docking_run_name)
                    failed_runs += [docking_run_name]
                except pebble.ProcessExpired as e:
                    logger.error("Docking failed for", docking_run_name)
                    logger.error(f"\t{e}. Exit code {e.exitcode}")
                    failed_runs += [docking_run_name]
                except Exception as e:
                    logger.error(
                        f"Docking failed for {docking_run_name}, with Exception: {e.__class__.__name__}"
                    )
                    if hasattr(e, "traceback"):
                        logger.error(e.traceback)
                    failed_runs += [docking_run_name]

                # things are going poorly, lets stop
                if len(failed_runs) > args.max_failures:
                    logger.critical(
                        f"CRITICAL: Too many failures ({len(failed_runs)}/{args.max_failures}) while running docking, aborting"
                    )
                    res.cancel()

            if args.verbose:
                logging.info(f"Docking complete with {len(failed_runs)} failures.")
                if len(failed_runs) > 0:
                    failed_run_str = "\n".join(failed_runs)
                    logger.error(f"Failed runs:\n{failed_run_str}\n")
            else:
                logging.info(
                    f"Docking complete with {len(failed_runs)} failures, use --verbose to see which ones."
                )

    else:
        logger.info("Running docking using single core this will take a while...")
        logger.info(f"Running {len(mp_args)} docking runs over 1 core.")
        logger.info("not using failure counter for single core")
        results_list = [mp_func_ml_applied(*args_list) for args_list in mp_args]

    logger.info("\nDocking complete!\n")
    logger.info("Writing results")

    logger.info(f"Docking finished for {len(results_list)} runs.")
    # Preparing results dataframe
    # TODO: convert these SD tags to live somewhere else
    results_cols = [
        "ligand_id",
        "du_structure",
        "docked_file",
        "pose_id",
        "docked_RMSD",
        "POSIT_prob",
        "POSIT_method",
        "chemgauss4_score",
        "clash",
        "SMILES",
        "GAT_score",
        "schnet_score",
    ]

    # results_list has the form [[(res1, res2, res3, ...)], [(res1, res2, res3, ...)], ...]
    # this flattens the list to look like [(res1, res2, res3, ...), (res1, res2, res3, ...), ...]
    # TODO: make this unnecessary?
    flattened_results_list = [res for res_list in results_list for res in res_list]
    results_df = pandas.DataFrame(flattened_results_list, columns=results_cols)

    # Save results to csv
    csv_name = f"{args.output_dir}/{log_name}-results.csv"
    results_df.to_csv(csv_name, index=False)
    logger.info(f"Saved results to {csv_name}")

    end = datetime.now().isoformat()
    logger.info(f"Started at {start}; finished at {end}")


if __name__ == "__main__":
    main()
