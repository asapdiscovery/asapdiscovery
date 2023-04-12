"""
Script to dock an SDF file of ligands to prepared structures.
"""
import argparse
import logging
import multiprocessing as mp
import os
import pickle as pkl
import re
import shutil
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

    # try:
    #     _ = load_openeye_sdf(os.path.join(d, "docked.sdf"))
    # except Exception:
    #     return False
    #
    # try:
    #     _ = pkl.load(open(os.path.join(d, "results.pkl"), "rb"))
    # except Exception:
    #     return False

    return True


def load_dus(file_base, by_compound=False):
    """
    Load all present oedu files. If `file_base` is a directory, os.walk will be
    used to find all .oedu files in the directory. Otherwise, it will be
    assessed with glob.

    Parameters
    ----------
    file_base : str
        Directory/base filepath for .oedu files, or best_results.csv file if
        `by_compound` is True.
    by_compound : bool, default=False
        Whether to load by dataset (False) or by compound_id (True).

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping Mpro dataset name/compound id to list of full
        Mpro names/compound ids (with chain)
    Dict[str, oechem.OEDesignUnit]
        Dictionary mapping full Mpro name/compound id (including chain) to its
        design unit
    """
    logger = logging.getLogger("run_docking_oe")

    if os.path.isdir(file_base):
        logger.info(f"Using {file_base} as directory")
        all_fns = [
            os.path.join(file_base, fn)
            for _, _, files in os.walk(file_base)
            for fn in files
            if fn[-4:] == "oedu"
        ]
    elif os.path.isfile(file_base) and by_compound:
        logger.info(f"Using {file_base} as file")
        df = pandas.read_csv(file_base)
        all_fns = [
            os.path.join(os.path.dirname(fn), "predocked.oedu")
            for fn in df["Docked_File"]
        ]
    else:
        logger.info(f"Using {file_base} as glob")
        all_fns = glob(file_base)

    # check that we actually have loaded in prepped receptors.
    check_filelist_has_elements(all_fns, tag="prepped receptors")

    du_dict = {}
    dataset_dict = {}
    if by_compound:
        re_pat = r"([A-Z]{3}-[A-Z]{3}-[a-z0-9]+-[0-9]+)_[0-9][A-Z]"
    else:
        re_pat = r"(Mpro-[A-Za-z][0-9]+)_[0-9][A-Z]"
    logger.info(f"Loading {len(all_fns)} design units")
    for fn in all_fns:
        m = re.search(re_pat, fn)
        if m is None:
            search_type = "compound_id" if by_compound else "Mpro dataset"
            logger.warning(f"No {search_type} found for {fn}")
            continue

        dataset = m.groups()[0]
        full_name = m.group()
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(fn, du):
            logger.error(f"Failed to read DesignUnit {fn}")
            continue
        du_dict[full_name] = du
        try:
            dataset_dict[dataset].append(full_name)
        except KeyError:
            dataset_dict[dataset] = [full_name]
    logger.info(f"{len(du_dict.keys())} design units loaded")
    return dataset_dict, du_dict


def mp_func(out_dir, lig_name, du_name, compound_name, *args, GAT_model=None, **kwargs):
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
    compound_name : str
        Compound name, used for error messages if given
    GAT_model : GATInference, optional
        GAT model to use for inference. If None, will not perform inference.

    Returns
    -------
    """
    logname = f"run_docking_oe.{compound_name}"

    before = datetime.now().isoformat()
    if check_results(out_dir):
        logger = FileLogger(logname, path=str(out_dir)).getLogger()
        logger.info(f"Found results for {compound_name}")
        after = datetime.now().isoformat()
        logger.info(f"Start: {before}, End: {after}")
        return
    else:
        os.makedirs(out_dir, exist_ok=True)
        logger = FileLogger(logname, path=str(out_dir)).getLogger()
        logger.info(f"No results for {compound_name} found, running docking")
        errfs = oechem.oeofstream(os.path.join(out_dir, f"openeye_{logname}-log.txt"))
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)
        oechem.OEThrow.Info(f"Starting docking for {logname}")

    success, posed_mol, docking_id = run_docking_oe(*args, **kwargs)
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
    return


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
        help=(
            "Path/glob to prepped receptor(s), or best_results.csv file if "
            "--by_compound is given."
        ),
    )
    parser.add_argument(
        "-s",
        "--sort_res",
        help="Pickle file giving compound_ids, xtal_ids, and sort_idxs.",
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
        default=30,
        help=(
            "Timeout (in seconds) for each docking thread. "
            "Set to a negative number to disable."
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
        action="store_true",
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

    # Parse symlinks in output_dir
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    logger = FileLogger(args.log_name, path=str(output_dir)).getLogger()
    start = datetime.now().isoformat()
    if args.exp_file:
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
    elif args.exp_file is None:
        raise ValueError("Need to specify exactly one of --exp_file or --lig_file.")
    n_mols = len(mols)

    # load ml models
    if args.gat:
        from asapdiscovery.ml.inference import GATInference  # noqa: E402

        GAT_model = GATInference("model1")
    else:
        GAT_model = None

    # Load all receptor DesignUnits
    dataset_dict, du_dict = load_dus(
        file_base=args.receptor, by_compound=args.by_compound
    )
    logger.info(f"{n_mols} molecules found")
    logger.info(f"{len(du_dict.keys())} receptor structures found")
    assert n_mols > 0
    assert len(du_dict.keys()) > 0

    # Load sort indices if given
    if args.sort_res:
        compound_ids, xtal_ids, sort_idxs = pkl.load(open(args.sort_res, "rb"))
        # If we're docking to all DUs, set top_n appropriately
        if args.top_n == -1:
            args.top_n = len(xtal_ids)

        # Make sure that compound_ids match with experimental data if that's
        #  what we're using
        if args.exp_file:
            assert all(
                [
                    compound_id == c.compound_id
                    for (compound_id, c) in zip(compound_ids, exp_compounds)
                ]
            ), (
                "Sort result compound_ids are not equivalent to "
                "compound_ids in --exp_file."
            )
    else:
        # Check to see if the SDF files have a Compound_ID Column
        if all(len(oechem.OEGetSDData(mol, "Compound_ID")) > 0 for mol in mols):
            logger.info("Using Compound_ID column from sdf file")
            compound_ids = [oechem.OEGetSDData(mol, "Compound_ID") for mol in mols]
        else:
            # Use index as compound_id
            compound_ids = [str(i) for i in range(n_mols)]
        # Get dataset values from DesignUnit filenames
        xtal_ids = list(dataset_dict.keys())
        # Arbitrary sort index, same for each ligand
        sort_idxs = [list(range(len(xtal_ids)))] * n_mols
        args.top_n = len(xtal_ids)

    # make multiprocessing args
    mp_args = []
    for i, m in enumerate(mols):
        dock_dus = []
        xtals = []
        for xtal in sort_idxs[i][: args.top_n]:
            if xtal_ids[xtal] not in dataset_dict:
                continue
            # Get the DU for each full Mpro name associated with this dataset
            dock_dus.extend([du_dict[x] for x in dataset_dict[xtal_ids[xtal]]])
            xtals.extend(dataset_dict[xtal_ids[xtal]])
        new_args = [
            (
                os.path.join(args.output_dir, f"{compound_ids[i]}_{x}"),
                compound_ids[i],
                x,
                f"{compound_ids[i]}_{x}",
                du,
                m,
                args.docking_sys.lower(),
                args.relax.lower(),
                args.hybrid,
                f"{compound_ids[i]}_{x}",
                args.omega,
                args.num_poses,
            )
            for du, x in zip(dock_dus, xtals)
        ]
        mp_args.extend(new_args)

    if args.debug_num > 0:
        mp_args = mp_args[: args.debug_num]

    # Apply ML arguments as kwargs to mp_func
    mp_func_ml_applied = partial(mp_func, GAT_model=GAT_model)

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
    if args.num_cores > 1:
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
            results_df = []
            # List to keep track of which runs failed
            failed_runs = []

            # TimeoutError is only raised when we try to access the result. Do things
            #  this way so we can keep track of which compound:xtals timed out
            res_iter = res.result()
            for args_list in mp_args:
                try:
                    cur_res = next(res_iter)
                    results_df += [cur_res]
                except StopIteration:
                    # We've reached the end of the results iterator so just break
                    break
                except TimeoutError:
                    # This compound:xtal combination timed out
                    print("Docking timed out for", args_list[8], flush=True)
                    failed_runs += [args_list[8]]
                except pebble.ProcessExpired as e:
                    print("Docking failed for", args_list[8], flush=True)
                    print(f"\t{e}. Exit code {e.exitcode}", flush=True)
                    failed_runs += [args_list[8]]
                except Exception as e:
                    print(
                        "Docking failed for",
                        args_list[8],
                        "with Exception",
                        e,
                        flush=True,
                    )
                    print(e.traceback, flush=True)
                    failed_runs += [args_list[8]]
            print(f"Docking failed for {len(failed_runs)} runs", flush=True)
    # Apply ML arguments as kwargs to mp_func
    mp_func_ml_applied = partial(mp_func, GAT_model=GAT_model)

    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    logger.info(f"CPUs: {mp.cpu_count()}")
    logger.info(f"N Processes: {len(mp_args)}")
    logger.info(f"N Cores: {args.num_cores}")

    mp_args = mp_args[: args.debug_num]
    logger.info(f"Running {len(mp_args)} docking runs over {nprocs} cores.")

    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(mp_func_ml_applied, mp_args)
    end = datetime.now().isoformat()
    logger.info(f"Started at {start}; finished at {end}")


if __name__ == "__main__":
    main()
