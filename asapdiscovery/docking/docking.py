def run_docking_oe(
    du, dock_lig, dock_sys, relax="none", hybrid=False, compound_name=None
):
    """
    Run docking using OpenEye. The returned OEGraphMol object will have the
    following SD tags set:
      * Docking_<docking_id>_RMSD: RMSD score to original molecule
      * Docking_<docking_id>_POSIT: POSIT probability
      * Docking_<docking_id>_Chemgauss4: Chemgauss4 score
      * Docking_<docking_id>_clash: clash results

    Parameters
    ----------
    du : oechem.OEDesignUnit
        DesignUnit receptor to dock to
    dock_lig : oechem.OEMol
        Mol object to dock
    dock_sys : str
        Which docking system to use ["posit", "hybrid"]
    relax : str, default="none"
        When to check for relaxation ["clash", "all", "none"]
    hybrid : bool, default=False
        Set POSIT methods to only use Hybrid
    compound_name : str, optional
        Compound name, used for error messages if given

    Returns
    -------
    bool
        If docking succeeded
    oechem.OEGraphMol
        Posed molecule (with set SD tags)
    str
        Generated docking_id, used to access SD tag data
    """
    from openeye import oechem, oedocking
    from asapdiscovery.docking.analysis import calculate_rmsd_openeye

    ## Convert to OEMol for docking purposes
    dock_lig = oechem.OEMol(dock_lig)

    ## Set docking string id (for SD tags)
    docking_id = [dock_sys]

    ## Keep track of if there's a clash (-1 if not using POSIT, 0 if no clash,
    ##  1 if there was a clash that couldn't be resolved)
    clash = -1

    ## Get ligand to dock
    if dock_sys == "posit":
        ## Set up POSIT docking options
        opts = oedocking.OEPositOptions()
        ## kinoml has the below option set, but the accompanying comment implies
        ##  that we should be ignoring N stereochemistry, which, paradoxically,
        ##  corresponds to a False option (the default)
        opts.SetIgnoreNitrogenStereo(True)
        ## Set the POSIT methods to only be hybrid (otherwise leave as default
        ##  of all)
        if hybrid:
            opts.SetPositMethods(oedocking.OEPositMethod_HYBRID)
            docking_id.append("hybrid")

        ## Set up pose relaxation
        if relax == "clash":
            clash = 0
            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_CLASHED)
        elif relax == "all":
            clash = 0
            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
        elif relax != "none":
            ## Don't need to do anything for none bc that's already the default
            raise ValueError(f'Unknown arg for relaxation "{relax}"')
        docking_id.append(relax)

        if compound_name:
            print(
                f"Running POSIT {'hybrid' if hybrid else 'all'} docking with "
                f"{relax} relaxation for {compound_name}",
                flush=True,
            )

        ## Set up poser object
        poser = oedocking.OEPosit(opts)
        poser.AddReceptor(du)

        ## Run posing
        pose_res = oedocking.OESinglePoseResult()
        try:
            ret_code = poser.Dock(pose_res, dock_lig)
        except TypeError as e:
            print(pose_res, dock_lig, type(dock_lig), flush=True)
            raise e
    elif dock_sys == "hybrid":
        if compound_name:
            print(f"Running Hybrid docking for {compound_name}", flush=True)

        ## Set up poser object
        poser = oedocking.OEHybrid()
        poser.Initialize(du)

        ## Run posing
        posed_mol = oechem.OEMol()
        ret_code = poser.DockMultiConformerMolecule(posed_mol, dock_lig)

        posit_prob = -1.0
    else:
        raise ValueError(f'Unknown docking system "{dock_sys}"')

    if ret_code == oedocking.OEDockingReturnCode_NoValidNonClashPoses:
        ## For POSIT with clash removal, if no non-clashing pose can be found,
        ##  re-run with no clash removal
        opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_NONE)
        clash = 1

        if compound_name:
            print(
                f"Re-running POSIT {'hybrid' if hybrid else 'all'} docking",
                f"with no relaxation for {compound_name}",
                flush=True,
            )

        ## Set up poser object
        poser = oedocking.OEPosit(opts)
        poser.AddReceptor(du)

        ## Run posing
        pose_res = oedocking.OESinglePoseResult()
        ret_code = poser.Dock(pose_res, dock_lig)

    ## Check results
    if ret_code == oedocking.OEDockingReturnCode_Success:
        if dock_sys == "posit":
            posed_mol = pose_res.GetPose()
            posit_prob = pose_res.GetProbability()

        ## Get the Chemgauss4 score (adapted from kinoml)
        pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
        pose_scorer.Initialize(du)
        chemgauss_score = pose_scorer.ScoreLigand(posed_mol)
    else:
        err_type = oedocking.OEDockingReturnCodeGetName(ret_code)
        if compound_name:
            print(
                f"Pose generation failed for {compound_name} ({err_type})",
                flush=True,
            )
        return False, None, None

    ## Calculate RMSD
    posed_copy = posed_mol.CreateCopy()

    rmsd = calculate_rmsd_openeye(dock_lig, posed_copy)

    ## Set SD tags for molecule
    docking_id = "_".join(docking_id)
    oechem.OESetSDData(posed_mol, f"Docking_{docking_id}_RMSD", str(rmsd))
    oechem.OESetSDData(
        posed_mol, f"Docking_{docking_id}_POSIT", str(posit_prob)
    )
    oechem.OESetSDData(
        posed_mol, f"Docking_{docking_id}_Chemgauss4", str(chemgauss_score)
    )
    oechem.OESetSDData(posed_mol, f"Docking_{docking_id}_clash", str(clash))
    oechem.OESetSDData(posed_mol, f"SMILES", oechem.OEMolToSmiles(dock_lig))

    ## Set molecule name if given
    if compound_name:
        posed_mol.SetTitle(compound_name)

    return True, oechem.OEGraphMol(posed_mol), docking_id
