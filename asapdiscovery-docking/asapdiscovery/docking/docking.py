import logging
import os
import pickle as pkl
from collections import namedtuple
from datetime import datetime
from pathlib import Path  # noqa: F401
from typing import List, Optional, Tuple  # noqa: F401

import numpy as np
import pandas as pd
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    load_openeye_sdf,
    oechem,
    oedocking,
    save_openeye_pdb,
    save_openeye_sdf,
)
from asapdiscovery.docking.analysis import calculate_rmsd_openeye
from asapdiscovery.modeling.modeling import split_openeye_design_unit

POSIT_METHODS = ("all", "hybrid", "fred", "mcs", "shapefit")

posit_methods = namedtuple("posit_methods", POSIT_METHODS)
posit_method_ints = posit_methods(
    oedocking.OEPositMethod_ALL,
    oedocking.OEPositMethod_HYBRID,
    oedocking.OEPositMethod_FRED,
    oedocking.OEPositMethod_MCS,
    oedocking.OEPositMethod_SHAPEFIT,
)


def run_docking_oe(
    du,
    orig_mol,
    dock_sys,
    relax="none",
    posit_method: str = "all",
    compound_name=None,
    use_omega=False,
    num_poses=1,
    log_name="run_docking_oe",
):
    """
    Run docking using OpenEye. The returned OEGraphMol object will have the
    following SD tags set:
      * Docking_<docking_id>_RMSD: RMSD score to original molecule
      * Docking_<docking_id>_POSIT: POSIT probability
      * Docking_<docking_id>_POSIT_method: POSIT method used in docking
      * Docking_<docking_id>_Chemgauss4: Chemgauss4 score
      * Docking_<docking_id>_clash: clash results

    Parameters
    ----------
    du : oechem.OEDesignUnit
        DesignUnit receptor to dock to
    orig_mol : oechem.OEMol
        Mol object to dock
    dock_sys : str
        Which docking system to use ["posit", "hybrid"]
    relax : str, default="none"
        When to check for relaxation ["clash", "all", "none"]
    posit_method : bool, default=False
        Set POSIT method to use one of the POSIT_METHODS
    log_name : str, optional
        Name of high-level logger to use
    compound_name : str, optional
        Compound name, used for error messages if given
    use_omega : bool, default=False
        Use OEOmega to manually generate conformations
    num_poses : int, default=1
        Number of poses to return from docking (only relevant for POSIT)

    Returns
    -------
    bool
        If docking succeeded
    oechem.OEMol
        Molecule with each pose as a different conformation
        (each with set SD tags)
    str
        Generated docking_id, used to access SD tag data
    """
    import sys

    if compound_name:
        logname = f"{log_name}.{compound_name}"
    else:
        logname = log_name
    logger = logging.getLogger(logname)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.warning(f"No logfile with name '{logname}' exists, using stdout instead")
    logger.info(f"Running docking for {compound_name}")

    oechem.OEThrow.Debug("Confirm that OE logging is working")

    # Make copy so we can keep the original for RMSD purposes
    orig_mol = orig_mol.CreateCopy()

    # Convert to OEMol for docking purposes
    dock_lig = oechem.OEMol(orig_mol.CreateCopy())

    # Perform OMEGA sampling
    if use_omega:
        from asapdiscovery.data.openeye import oeomega

        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)
        omega = oeomega.OEOmega(omegaOpts)
        ret_code = omega.Build(dock_lig)
        if ret_code:
            logger.error(f"Omega failed with error {oeomega.OEGetOmegaError(ret_code)}")

    # Set docking string id (for SD tags)
    docking_id = [dock_sys]

    # Keep track of if there's a clash (-1 if not using POSIT, 0 if no clash,
    #  1 if there was a clash that couldn't be resolved)
    clash = -1

    # Get ligand to dock
    if dock_sys == "posit":
        # Set up POSIT docking options
        opts = oedocking.OEPositOptions()
        # kinoml has the below option set, but the accompanying comment implies
        #  that we should be ignoring N stereochemistry, which, paradoxically,
        #  corresponds to a False option (the default)
        opts.SetIgnoreNitrogenStereo(True)
        # Set the POSIT methods to only be hybrid (otherwise leave as default
        #  of all)
        if posit_method not in POSIT_METHODS:
            raise ValueError(
                f"Unknown POSIT method {posit_method}. Must be one of {POSIT_METHODS}"
            )
        else:
            method = posit_method_ints._asdict()[posit_method]
            opts.SetPositMethods(method)
            docking_id.append(posit_method)

        # Set up pose relaxation
        if relax == "clash":
            clash = 0
            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_CLASHED)
        elif relax == "all":
            clash = 0
            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
        elif relax != "none":
            # Don't need to do anything for none bc that's already the default
            raise ValueError(f'Unknown arg for relaxation "{relax}"')
        docking_id.append(relax)

        if compound_name:
            logger.info(
                f"Running POSIT method '{posit_method}' docking with "
                f"{relax} relaxation for {compound_name}"
            )

        # Set up poser object
        poser = oedocking.OEPosit(opts)
        if not poser.AddReceptor(du):
            logger.critical("Failed to add receptor to POSIT object")
            raise RuntimeError("Failed to add receptor to POSIT object")

        # Run posing
        pose_res = oedocking.OEPositResults()
        try:
            ret_code = poser.Dock(pose_res, dock_lig, num_poses)
        except TypeError as e:
            logger.error(pose_res, dock_lig, type(dock_lig))
            raise e
    elif dock_sys == "hybrid":
        if compound_name:
            logger.info(f"Running Hybrid docking for {compound_name}")

        # Set up poser object
        poser = oedocking.OEHybrid()

        # Ensure poser is initialized
        if not poser.Initialize(du):
            logger.critical("Failed to add receptor to HYBRID object")
            raise RuntimeError("Failed to add receptor to HYBRID object")

        # Run posing
        posed_mol = oechem.OEMol()
        ret_code = poser.DockMultiConformerMolecule(posed_mol, dock_lig)

        # Place in list to match output
        posed_mols = [posed_mol]

        posit_probs = [-1.0]
        posit_methods = ["NA"]
    else:
        raise ValueError(f'Unknown docking system "{dock_sys}"')

    if ret_code == oedocking.OEDockingReturnCode_NoValidNonClashPoses:
        # For POSIT with clash removal, if no non-clashing pose can be found,
        #  re-run with no clash removal
        opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_NONE)
        clash = 1

        if compound_name:
            logger.info(
                f"Re-running POSIT with method '{posit_method}' docking with no relaxation for {compound_name}",
            )

        # Set up poser object
        poser = oedocking.OEPosit(opts)
        poser.AddReceptor(du)

        # Run posing
        pose_res = oedocking.OEPositResults()
        ret_code = poser.Dock(pose_res, dock_lig, num_poses)

    # Check results
    if ret_code == oedocking.OEDockingReturnCode_Success and dock_sys == "posit":
        posed_mols = []
        posit_probs = []
        posit_methods = []
        for single_res in pose_res.GetSinglePoseResults():
            posed_mols.append(single_res.GetPose())
            posit_probs.append(single_res.GetProbability())
            posit_methods.append(
                oedocking.OEPositMethodGetName(single_res.GetPositMethod())
            )
    else:
        err_type = oedocking.OEDockingReturnCodeGetName(ret_code)
        if compound_name:
            logger.error(
                f"Pose generation failed for {compound_name} ({err_type})",
            )
        return False, None, None

    # Set docking_id key for SD tags
    docking_id = "_".join(docking_id)

    for i, (mol, prob, method) in enumerate(
        zip(posed_mols, posit_probs, posit_methods)
    ):
        # Get the Chemgauss4 score (adapted from kinoml)
        pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
        pose_scorer.Initialize(du)
        chemgauss_score = pose_scorer.ScoreLigand(mol)

        # Calculate RMSD
        posed_copy = mol.CreateCopy()
        rmsd = calculate_rmsd_openeye(orig_mol.CreateCopy(), posed_copy)

        # First copy over original SD tags
        for sd_data in oechem.OEGetSDDataPairs(orig_mol):
            oechem.OESetSDData(mol, sd_data)

        # Set SD tags for molecule
        oechem.OESetSDData(mol, f"Docking_{docking_id}_RMSD", str(rmsd))
        oechem.OESetSDData(mol, f"Docking_{docking_id}_POSIT", str(prob))
        oechem.OESetSDData(mol, f"Docking_{docking_id}_POSIT_method", method)
        oechem.OESetSDData(
            mol, f"Docking_{docking_id}_Chemgauss4", str(chemgauss_score)
        )
        oechem.OESetSDData(mol, f"Docking_{docking_id}_clash", str(clash))
        oechem.OESetSDData(mol, "SMILES", oechem.OEMolToSmiles(mol))

        # Set molecule name if given
        if compound_name:
            mol.SetTitle(f"{compound_name}_{i}")

    # Combine all the conformations into one
    combined_mol = oechem.OEMol(posed_mols[0])
    for mol in posed_mols[1:]:
        combined_mol.NewConf(mol)
    assert combined_mol.NumConfs() == len(posed_mols)

    return True, combined_mol, docking_id


def docking_result_cols() -> list[str]:
    return [
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
        "SCHNET_score",
    ]


def make_docking_result_dataframe(
    results: list,
    output_dir: Path,
    save_csv: bool = True,
    results_cols: Optional[list[str]] = docking_result_cols(),
    csv_name: Optional[str] = "results.csv",
) -> tuple[pd.DataFrame, Path]:
    """
    Save results to a CSV file

    Parameters
    ----------
    results : List
        List of results from docking
    output_dir : Path
        Path to output directory
    results_cols : Optional[List[str]], optional
        List of column names for results, by default will use a set of hardcoded column names
    csv_name : Optional[str], optional
        Name of CSV file, by default "results.csv"

    Returns
    -------
    pandas.DataFrame
        DataFrame of results
    Path
        Path to CSV file
    """
    _results_cols = results_cols

    flattened_results_list = [res for res_list in results for res in res_list]
    results_df = pd.DataFrame(flattened_results_list, columns=_results_cols)
    if save_csv:
        csv = output_dir / csv_name
        results_df.to_csv(csv, index=False)
    else:
        csv = None
    return results_df, csv


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


def dock_and_score_pose_oe(
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
        # this interferes with OEOmega see https://github.com/openforcefield/openff-toolkit/issues/1615
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
    errfs.close()
    logger.info(f"Start: {before}, End: {after}")
    return results
