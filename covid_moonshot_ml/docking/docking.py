import os

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


def parse_xtal(x_fn, x_dir):
    """
    Load all crystal structures into schema.CrystalCompoundData objects.
    Parameters
    ----------
    x_fn : str
        CSV file giving information on each crystal structure
    x_dir : str
        Path to directory containing directories with crystal structure PDB
        files
    Returns
    -------
    List[schema.CrystalCompoundData]
        List of parsed crystal structures
    """
    import pandas

    from ..schema import CrystalCompoundData

    df = pandas.read_csv(x_fn)

    ## Find all P-files
    idx = [(type(d) is str) and ("-P" in d) for d in df["Dataset"]]

    ## Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [
        dict(zip(("smiles", "dataset", "compound_id"), r[1].values))
        for r in df.loc[idx, ["SMILES", "Dataset", "Compound ID"]].iterrows()
    ]

    ## Add structure filename information
    for d in xtal_dicts:
        fn_base = f'{x_dir}/{d["dataset"]}_0{{}}/{d["dataset"]}_0{{}}_{{}}.pdb'
        for suf in ["seqres", "bound"]:
            for chain in ["A", "B"]:
                fn = fn_base.format(chain, chain, suf)
                if os.path.isfile(fn):
                    d["str_fn"] = fn
                    break
            if os.path.isfile(fn):
                break
        assert os.path.isfile(fn), f'No structure found for {d["dataset"]}.'

    ## Build CrystalCompoundData objects for each row
    xtal_compounds = [CrystalCompoundData(**d) for d in xtal_dicts]

    return xtal_compounds


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


def run_docking_oe(du, dock_lig, dock_sys, relax="none", hybrid=False):
    """
    Run docking using OpenEye.

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

    Returns
    -------
    bool
        If docking succeeded
    float
        RMSD
    float
        POSIT prob
    float
        Chemgauss4 score
    int
        Whether clash correction was disabled, enabled and no clash was found,
        or enables and clash was found ([-1, 0, 1] respectively)
    """
    from openeye import oechem, oedocking

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

        ## Set up poser object
        poser = oedocking.OEPosit(opts)
        poser.AddReceptor(du)

        ## Run posing
        pose_res = oedocking.OESinglePoseResult()
        ret_code = poser.Dock(pose_res, dock_lig)
    elif dock_sys == "hybrid":
        print("Running Hybrid docking", flush=True)

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

        print(
            f"Re-running POSIT {'hybrid' if hybrid else 'all'} docking with "
            f"no relaxation for {lig_name}/{apo_name}",
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
        print(
            f"Pose generation failed for {lig_name}/{apo_name} ({err_type})",
            flush=True,
        )
        return False, -1.0, -1.0, -1.0, clash

    ## Calculate RMSD
    oechem.OECanonicalOrderAtoms(dock_lig)
    oechem.OECanonicalOrderBonds(dock_lig)
    oechem.OECanonicalOrderAtoms(posed_mol)
    oechem.OECanonicalOrderBonds(posed_mol)
    ## Get coordinates, filtering out Hs
    predocked_coords = [
        c
        for a in dock_lig.GetAtoms()
        for c in dock_lig.GetCoords()[a.GetIdx()]
        if a.GetAtomicNum() != 1
    ]
    docked_coords = [
        c
        for a in posed_mol.GetAtoms()
        for c in posed_mol.GetCoords()[a.GetIdx()]
        if a.GetAtomicNum() != 1
    ]
    rmsd = oechem.OERMSD(
        oechem.OEDoubleArray(predocked_coords),
        oechem.OEDoubleArray(docked_coords),
        len(predocked_coords) // 3,
    )

    return True, rmsd, posit_prob, chemgauss_score, clash
