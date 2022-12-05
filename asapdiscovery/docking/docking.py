from kinoml.core.proteins import Protein
from kinoml.core.ligands import Ligand
from kinoml.core.systems import ProteinLigandComplex


def build_docking_systems(
    exp_compounds, xtal_compounds, compound_idxs, n_top=1
):
    """
    Build systems to run through docking.
    Parameters
    ----------
    exp_compounds : list[schema.ExperimentalCompoundData]
        List of compounds to dock
    xtal_compounds : list[schema.CrystalCompoundData]
        List of all crystal structures
    compound_idxs : list[int]
        List giving the index of the crystal structure to dock to for each
        ligand. Should be the same length as `exp_compounds`
    n_top : int, default=1
        Dock to top `n_top` crystal structures

    Returns
    -------
    List[kinoml.core.systems.ProteinLigandComplex]
        List of protein+ligand systems for docking
    """
    systems = []
    for (c, idx) in zip(exp_compounds, compound_idxs):
        ## Make sure that there are enough crystal structures to dock to
        n_dock = min(n_top, len(idx))
        for i in range(n_dock):
            ## Build protein, ligand, and complex objects
            x = xtal_compounds[idx[i]]
            protein = Protein.from_file(x.str_fn, name="MPRO")
            protein.chain_id = x.str_fn.split("_")[-2][-1]
            protein.expo_id = "LIG"
            ligand = Ligand.from_smiles(smiles=c.smiles, name=c.compound_id)
            systems.append(ProteinLigandComplex(components=[protein, ligand]))

    return systems


def build_docking_system_direct(prot_mol, lig_smi, prot_name, lig_name):
    """
    Build system to run through kinoml docking from OEGraphMol objects.

    Parameters
    ----------
    prot_mol : oechem.OEGraphMol
        Protein molecule.
    lig_smi : str
        Ligand SMILES string.
    prot_name : str
        Name of protein.
    lig_name : str
        Name of ligand.

    Returns
    -------
    kinoml.core.systems.ProteinLigandComplex
    """
    protein = Protein(molecule=prot_mol, name=prot_name)
    ligand = Ligand.from_smiles(smiles=lig_smi, name=lig_name)

    return ProteinLigandComplex(components=[protein, ligand])


def build_combined_protein_system_from_sdf(pdb_fn, sdf_fn):
    protein = Protein.from_file(pdb_fn, name="MERS-Mpro")
    ligand = Ligand.from_file(sdf_fn)
    return ProteinLigandComplex


def run_docking(cache_dir, output_dir, loop_db, n_procs, docking_systems):
    from kinoml.features.complexes import OEDockingFeaturizer

    featurizer = OEDockingFeaturizer(
        cache_dir=cache_dir,
        output_dir=output_dir,
        loop_db=loop_db,
        n_processes=n_procs,
    )
    docking_systems = featurizer.featurize(docking_systems)

    return docking_systems


def run_docking_oe(
    du,
    orig_mol,
    dock_sys,
    relax="none",
    hybrid=False,
    compound_name=None,
    use_omega=False,
    num_poses=1,
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
    from openeye import oechem, oedocking
    from asapdiscovery.docking.analysis import calculate_rmsd_openeye

    ## Make copy so we can keep the original for RMSD purposes
    orig_mol = orig_mol.CreateCopy()

    ## Convert to OEMol for docking purposes
    dock_lig = oechem.OEMol(orig_mol.CreateCopy())

    ## Perform OMEGA sampling
    if use_omega:
        from openeye import oeomega

        omega = oeomega.OEOmega()
        ret_code = omega.Build(dock_lig)
        if ret_code:
            print(
                f"Omega failed with error {oeomega.OEGetOmegaError(ret_code)}",
                flush=True,
            )

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
        pose_res = oedocking.OEPositResults()
        try:
            ret_code = poser.Dock(pose_res, dock_lig, num_poses)
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

        ## Place in list to match output
        posed_mols = [posed_mol]

        posit_probs = [-1.0]
        posit_methods = ["NA"]
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
        pose_res = oedocking.OEPositResults()
        ret_code = poser.Dock(pose_res, dock_lig, num_poses)

    ## Check results
    if (
        ret_code == oedocking.OEDockingReturnCode_Success
        and dock_sys == "posit"
    ):
        posed_mols = []
        posit_probs = []
        posit_methods = []
        for r in pose_res.GetSinglePoseResults():
            posed_mols.append(r.GetPose())
            posit_probs.append(r.GetProbability())
            posit_methods.append(
                oedocking.OEPositMethodGetName(r.GetPositMethod())
            )
    else:
        err_type = oedocking.OEDockingReturnCodeGetName(ret_code)
        if compound_name:
            print(
                f"Pose generation failed for {compound_name} ({err_type})",
                flush=True,
            )
        return False, None, None

    ## Set docking_id key for SD tags
    docking_id = "_".join(docking_id)

    for i, (mol, prob, method) in enumerate(
        zip(posed_mols, posit_probs, posit_methods)
    ):
        ## Get the Chemgauss4 score (adapted from kinoml)
        pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
        pose_scorer.Initialize(du)
        chemgauss_score = pose_scorer.ScoreLigand(mol)

        ## Calculate RMSD
        posed_copy = mol.CreateCopy()
        rmsd = calculate_rmsd_openeye(orig_mol.CreateCopy(), posed_copy)

        ## First copy over original SD tags
        for sd_data in oechem.OEGetSDDataPairs(orig_mol):
            oechem.OESetSDData(mol, sd_data)

        ## Set SD tags for molecule
        oechem.OESetSDData(mol, f"Docking_{docking_id}_RMSD", str(rmsd))
        oechem.OESetSDData(mol, f"Docking_{docking_id}_POSIT", str(prob))
        oechem.OESetSDData(mol, f"Docking_{docking_id}_POSIT_method", method)
        oechem.OESetSDData(
            mol, f"Docking_{docking_id}_Chemgauss4", str(chemgauss_score)
        )
        oechem.OESetSDData(mol, f"Docking_{docking_id}_clash", str(clash))
        oechem.OESetSDData(mol, f"SMILES", oechem.OEMolToSmiles(mol))

        ## Set molecule name if given
        if compound_name:
            mol.SetTitle(f"{compound_name}_{i}")

    ## Combine all the conformations into one
    combined_mol = oechem.OEMol(posed_mols[0])
    for mol in posed_mols[1:]:
        combined_mol.NewConf(mol)
    assert combined_mol.NumConfs() == len(posed_mols)

    return True, combined_mol, docking_id
