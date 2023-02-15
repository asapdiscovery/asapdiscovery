import numpy as np


def rank_structures_openeye(
    exp_smi,
    exp_id,
    search_smis,
    search_ids,
    smi_conv=None,
    str_based=False,
    out_fn=None,
    n_draw=0,
):
    """
    Rank all molecules in search_mols based on their MCS with exp_mol.

    Parameters
    ----------
    exp_smi : str
        SMILES string of the experimental compound
    exp_id : str
        CDD compound ID of the experimental compound
    search_smis : list[str]
        List of SMILES of the ligands in the crystal compounds
    search_ids : list[str]
        List of IDs of crystal compounds
    smi_conv : function
        Function to convert a SMILES string to oechem.OEGraphMol
    str_based : bool, default=False
        Whether to use a structure-based search (True) or a more strict
        element-based search (False).
    out_fn : str, optional
        If not None, the prefix to save overlap molecule structure drawings.
    n_draw : int, optional
        Draw top n_draw matched molecules

    Returns
    -------
    numpy.ndarray
        Index that sorts `sort_smis` by decreasing similarity with `exp_smi`
        based on MCS search.
    """
    from openeye import oechem, oedepict

    if smi_conv is None:
        smi_conv = _smi_conv_oe

    if str_based:
        """
        For structure based matching
        Options for atom matching:
          * Aromaticity
          * HvyDegree - # heavy atoms bonded to
          * RingMember
        Options for bond matching:
          * Aromaticity
          * BondOrder
          * RingMember
        """
        atomexpr = (
            oechem.OEExprOpts_Aromaticity
            | oechem.OEExprOpts_HvyDegree
            | oechem.OEExprOpts_RingMember
        )
        bondexpr = (
            oechem.OEExprOpts_Aromaticity
            | oechem.OEExprOpts_BondOrder
            | oechem.OEExprOpts_RingMember
        )
    else:
        """
        For atom based matching
        Options for atom matching (predefined AutomorphAtoms):
          * AtomicNumber
          * Aromaticity
          * RingMember
          * HvyDegree - # heavy atoms bonded to
        Options for bond matching:
          * Aromaticity
          * BondOrder
          * RingMember
        """
        atomexpr = oechem.OEExprOpts_AutomorphAtoms
        bondexpr = (
            oechem.OEExprOpts_Aromaticity
            | oechem.OEExprOpts_BondOrder
            | oechem.OEExprOpts_RingMember
        )

    ## Set up the search pattern and MCS objects
    exp_mol = smi_conv(exp_smi)
    pattern_query = oechem.OEQMol(exp_mol)
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())

    ## Prepare exp_mol for drawing
    oedepict.OEPrepareDepiction(exp_mol)

    sort_args = []
    for smi in search_smis:
        mol = smi_conv(smi)

        ## MCS search
        mcs = next(iter(mcss.Match(mol, True)))
        sort_args.append((mcs.NumBonds(), mcs.NumAtoms()))

    sort_args = np.asarray(sort_args)
    sort_idx = np.lexsort(-sort_args.T)

    ## Find all substructure matching atoms and draw the molecule with those
    ##  atoms highlighted
    if out_fn is not None:
        for i in range(min(n_draw, len(search_smis))):
            mol_idx = sort_idx[i]
            smi = search_smis[mol_idx]
            mol = smi_conv(smi)

            ## Set up xtal mol for drawing
            oedepict.OEPrepareDepiction(mol)

            ## Set up aligned image
            alignres = oedepict.OEPrepareAlignedDepiction(mol, mcss)
            image = oedepict.OEImage(400, 200)
            grid = oedepict.OEImageGrid(image, 1, 2)
            opts = oedepict.OE2DMolDisplayOptions(
                grid.GetCellWidth(),
                grid.GetCellHeight(),
                oedepict.OEScale_AutoScale,
            )
            opts.SetTitleLocation(oedepict.OETitleLocation_Hidden)
            exp_scale = oedepict.OEGetMoleculeScale(exp_mol, opts)
            search_scale = oedepict.OEGetMoleculeScale(mol, opts)
            opts.SetScale(min(exp_scale, search_scale))
            exp_disp = oedepict.OE2DMolDisplay(mcss.GetPattern(), opts)
            search_disp = oedepict.OE2DMolDisplay(mol, opts)

            if alignres.IsValid():
                exp_abset = oechem.OEAtomBondSet(
                    alignres.GetPatternAtoms(), alignres.GetPatternBonds()
                )
                oedepict.OEAddHighlighting(
                    exp_disp,
                    oechem.OEBlueTint,
                    oedepict.OEHighlightStyle_BallAndStick,
                    exp_abset,
                )

                search_abset = oechem.OEAtomBondSet(
                    alignres.GetTargetAtoms(), alignres.GetTargetBonds()
                )
                oedepict.OEAddHighlighting(
                    search_disp,
                    oechem.OEBlueTint,
                    oedepict.OEHighlightStyle_BallAndStick,
                    search_abset,
                )

            exp_cell = grid.GetCell(1, 1)
            oedepict.OERenderMolecule(exp_cell, exp_disp)

            search_cell = grid.GetCell(1, 2)
            oedepict.OERenderMolecule(search_cell, search_disp)

            oedepict.OEWriteImage(
                f"{out_fn}_{search_ids[mol_idx]}_{i}.png", image
            )

    return sort_idx


def rank_structures_rdkit(
    exp_smi,
    exp_id,
    search_smis,
    search_ids,
    smi_conv=None,
    str_based=False,
    out_fn=None,
    n_draw=0,
):
    """
    Rank all molecules in search_mols based on their MCS with exp_mol.

    Parameters
    ----------
    exp_smi : str
        SMILES string of the experimental compound
    exp_id : str
        CDD compound ID of the experimental compound
    search_smis : list[str]
        List of SMILES of the ligands in the crystal compounds
    search_ids : list[str]
        List of IDs of crystal compounds
    smi_conv : function
        Function to convert a SMILES string to rdkit.Molecule
    str_based : bool, default=False
        Whether to use a structure-based search (True) or a more strict
        element-based search (False).
    out_fn : str, optional
        If not None, the prefix to save overlap molecule structure drawings.
    n_draw : int, optional
        Draw top n_draw matched molecules

    Returns
    -------
    numpy.ndarray
        Index that sorts `sort_smis` by decreasing similarity with `exp_smi`
        based on MCS search.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS, Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    if smi_conv is None:
        smi_conv = _smi_conv_rdkit

    if str_based:
        atom_compare = rdFMCS.AtomCompare.CompareAny
    else:
        atom_compare = rdFMCS.AtomCompare.CompareElements

    exp_mol = smi_conv(exp_smi)

    sort_args = []
    mcs_smarts = []
    for smi in search_smis:
        ## Convert SMILES to molecule
        mol = smi_conv(smi)

        ## Perform MCS search for each search molecule
        # maximize atoms first and then bonds
        # ensure that all ring bonds match other ring bonds and that all rings
        #  must be complete (allowing for incomplete rings causes problems
        #  for some reason)
        mcs = rdFMCS.FindMCS(
            [exp_mol, mol],
            maximizeBonds=False,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=atom_compare,
        )
        # put bonds before atoms because lexsort works backwards
        sort_args.append((mcs.numBonds, mcs.numAtoms))
        mcs_smarts.append(mcs.smartsString)

    sort_args = np.asarray(sort_args)
    sort_idx = np.lexsort(-sort_args.T)

    ## Find all substructure matching atoms and draw the molecule with those
    ##  atoms highlighted
    if out_fn is not None:
        for i in range(min(n_draw, len(search_smis))):
            mol_idx = sort_idx[i]
            smi = search_smis[mol_idx]
            mol = smi_conv(smi)

            patt = Chem.MolFromSmarts(mcs_smarts[mol_idx])
            hit_ats = list(mol.GetSubstructMatch(patt))
            hit_bonds = []
            for bond in patt.GetBonds():
                try:
                    aid1 = hit_ats[bond.GetBeginAtomIdx()]
                    aid2 = hit_ats[bond.GetEndAtomIdx()]
                except IndexError as e:
                    print(
                        len(hit_ats),
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        flush=True,
                    )
                    print(sort_args[mol_idx], flush=True)
                    print(search_ids[mol_idx], exp_id, flush=True)
                    print(
                        Chem.MolToSmiles(exp_mol),
                        Chem.MolToSmiles(mol),
                        flush=True,
                    )
                    print(i, mcs_smarts[mol_idx], flush=True)
                    raise e
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())

            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, highlightAtoms=hit_ats, highlightBonds=hit_bonds
            )
            d.FinishDrawing()
            d.WriteDrawingText(f"{out_fn}_{search_ids[mol_idx]}_{i}.png")

        Draw.MolToFile(exp_mol, f"{out_fn}.png")

    return sort_idx


def _smi_conv_rdkit(s):
    from rdkit import Chem

    return Chem.MolFromSmiles(Chem.CanonSmiles(s))


def _smi_conv_oe(s):
    from openeye import oechem

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, s)
    return mol
