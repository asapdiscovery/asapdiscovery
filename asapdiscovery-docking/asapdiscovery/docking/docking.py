import logging


def run_docking_oe(
    du,
    orig_mol,
    dock_sys,
    relax="none",
    hybrid=False,
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
    hybrid : bool, default=False
        Set POSIT methods to only use Hybrid
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
    from asapdiscovery.data.openeye import oechem, oedocking
    from asapdiscovery.docking.analysis import calculate_rmsd_openeye

    oechem.OEThrow.Debug("Confirm that OE logging is working")

    # Make copy so we can keep the original for RMSD purposes
    orig_mol = orig_mol.CreateCopy()

    # Convert to OEMol for docking purposes
    dock_lig = oechem.OEMol(orig_mol.CreateCopy())

    # Perform OMEGA sampling
    if use_omega:
        from openeye import oeomega

        omega = oeomega.OEOmega()
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
        if hybrid:
            opts.SetPositMethods(oedocking.OEPositMethod_HYBRID)
            docking_id.append("hybrid")

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
                f"Running POSIT {'hybrid' if hybrid else 'all'} docking with "
                f"{relax} relaxation for {compound_name}"
            )

        # Set up poser object
        poser = oedocking.OEPosit(opts)
        poser.AddReceptor(du)

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
        poser.Initialize(du)

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
                f"Re-running POSIT {'hybrid' if hybrid else 'all'} docking with no relaxation for {compound_name}",
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
