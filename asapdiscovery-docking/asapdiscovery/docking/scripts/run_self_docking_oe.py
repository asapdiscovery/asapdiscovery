"""

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
from asapdiscovery.data.openeye import load_openeye_sdf  # noqa: E402
from asapdiscovery.data.openeye import save_openeye_sdf  # noqa: E402
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate  # noqa: E402
from asapdiscovery.data.utils import check_filelist_has_elements  # noqa: E402
from asapdiscovery.docking.docking import run_docking_oe  # noqa: E402


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


def mp_func(
    out_dir,
    lig_name,
    du_name,
    log_name,
    complex_name,
    run_docking_oe_kwargs,
    GAT_model=None,
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
    complex_name : str
        Compound name, used for error messages if given
    GAT_model : GATInference, optional
        GAT model to use for inference. If None, will not perform inference.

    Returns
    -------
    """
    logname = f"{log_name}.{complex_name}"

    before = datetime.now().isoformat()
    if check_results(out_dir):
        logger = FileLogger(logname, path=str(out_dir)).getLogger()
        logger.info(f"Found results for {complex_name}")
        after = datetime.now().isoformat()
        results = pkl.load(open(os.path.join(out_dir, "results.pkl"), "rb"))
        logger.info(f"Start: {before}, End: {after}")
        return results
    else:
        os.makedirs(out_dir, exist_ok=True)
        logger = FileLogger(logname, path=str(out_dir)).getLogger()
        logger.info(f"No results for {complex_name} found, running docking")
        errfs = oechem.oeofstream(os.path.join(out_dir, f"openeye_{logname}-log.txt"))
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)
        oechem.OEThrow.Info(f"Starting docking for {logname}")

    success, posed_mol, docking_id = run_docking_oe(**run_docking_oe_kwargs)
    if success:
        out_fn = os.path.join(out_dir, "docked.sdf")
        save_openeye_sdf(posed_mol, out_fn)

        rmsds = []
        posit_probs = []
        posit_methods = []
        chemgauss_scores = []

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
        )
        for i, (rmsd, prob, method, chemgauss) in enumerate(
            zip(rmsds, posit_probs, posit_methods, chemgauss_scores)
        )
    ]

    pkl.dump(results, open(os.path.join(out_dir, "results.pkl"), "wb"))
    after = datetime.now().isoformat()
    logger.info(f"Start: {before}, End: {after}")
    return results


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-r",
        "--receptors",
        required=True,
        type=str,
        help="Path/glob to prepped receptor(s)",
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
        "--hybrid",
        type=bool,
        default=True,
        help="Whether to only use hybrid docking protocol in POSIT.",
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
        default=-1,
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
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logger = FileLogger("run_docking_oe", path=str(output_dir)).getLogger()
    logger.info(f"Output directory: {output_dir}")
    start = datetime.now().isoformat()
    logger.info(f"Starting run_docking_oe at {start}")

    # Set up ML model
    gat_model_string = "asapdiscovery-GAT-2023.04.12"
    if args.gat:
        from asapdiscovery.ml.inference import GATInference  # noqa: E402

        GAT_model = GATInference(gat_model_string)
        logger.info(f"Using GAT model: {gat_model_string}")
    else:
        logger.info("Skipping GAT model scoring")
        GAT_model = None

    # Load all receptor DesignUnits
    logger.info(f"Loading receptor DesignUnits from '{args.receptors}'")

    du_fns = glob(args.receptors)
    check_filelist_has_elements(du_fns, "receptors")

    mp_kwargs_list = []

    for fn in du_fns:
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(fn, du):
            logger.error(f"Failed to read DesignUnit {fn}")
            continue
        if not du.HasReceptor():
            logger.error(f"DesignUnit {fn} has no receptor")
            continue
        complex_name = du.GetTitle()

        lig = oechem.OEGraphMol()
        if not du.GetLigand(lig):
            logger.error(f"DesignUnit {fn} has no ligand")
            continue

        prot = oechem.OEGraphMol()
        du.GetProtein(prot)

        mp_args = {}
        mp_args["out_dir"] = output_dir / complex_name
        mp_args["lig_name"] = lig.GetTitle()
        mp_args["du_name"] = prot.GetTitle()
        mp_args["log_name"] = log_name
        mp_args["complex_name"] = complex_name
        mp_args["GAT_model"] = GAT_model

        run_docking_oe_args = {}
        run_docking_oe_args["du"] = du
        run_docking_oe_args["orig_mol"] = lig
        run_docking_oe_args["dock_sys"] = args.docking_sys
        run_docking_oe_args["relax"] = args.relax
        run_docking_oe_args["hybrid"] = args.hybrid
        run_docking_oe_args["use_omega"] = args.omega
        run_docking_oe_args["num_poses"] = args.num_poses
        run_docking_oe_args["log_name"] = log_name

        mp_args["run_docking_oe_kwargs"] = run_docking_oe_args

        mp_kwargs_list.append(mp_args)

    logger.info(f"{len(mp_kwargs_list)} receptor structures found and processed")

    if not len(mp_kwargs_list) > 0:
        raise ValueError("No docking runs were able to be prepared")

    # make multiprocessing args
    logger.info("Making multiprocessing args")

    if args.debug_num > 0:
        logger.info(f"DEBUG MODE: Only running {args.debug_num} docking runs")
        mp_kwargs_list = mp_kwargs_list[: args.debug_num]
    elif args.debug_num == 0:
        logger.info("DEBUG MODE: Skipping docking runs")
        mp_kwargs_list = []

    # Apply ML arguments as kwargs to mp_func
    mp_func_ml_applied = partial(mp_func, GAT_model=GAT_model)

    if args.num_cores > 1:
        logger.info("Running docking using multiprocessing")
        # reset failures
        logging.info(f"max_failures for running docking using MP : {args.max_failures}")

        nprocs = min(mp.cpu_count(), len(mp_kwargs_list), args.num_cores)
        logger.info(f"CPUs: {mp.cpu_count()}")
        logger.info(f"N Processes: {len(mp_kwargs_list)}")
        logger.info(f"N Cores: {args.num_cores}")
        logger.info(f"Running {len(mp_kwargs_list)} docking runs over {nprocs} cores.")
        with pebble.ProcessPool(max_workers=nprocs) as pool:
            if args.timeout <= 0:
                args.timeout = None
            # Need to flip args structure for pebble
            res = pool.map(
                mp_func_ml_applied, *zip(*mp_kwargs_list), timeout=args.timeout
            )

            # List to keep track of successful results
            results_list = []
            # List to keep track of which runs failed
            failed_runs = []

            # TimeoutError is only raised when we try to access the result. Do things
            #  this way so we can keep track of which compound:xtals timed out
            res_iter = res.result()
            for args_list in mp_kwargs_list:
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
        logger.info(f"Running {len(mp_kwargs_list)} docking runs over 1 core.")
        logger.info("not using failure counter for single core")
        results_list = [mp_func_ml_applied(**mp_kwargs) for mp_kwargs in mp_kwargs_list]

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
