from openeye import oechem, oedocking, oespruce

from .datasets.utils import (
    load_openeye_pdb,
    load_openeye_sdf,
    split_openeye_mol,
)


def du_to_complex(du, include_solvent=False):
    """
    Convert OEDesignUnit to OEGraphMol containing the protein and ligand from
    `du`.

    Parameters
    ----------
    du : oechem.OEDesignUnit
        OEDesignUnit object to extract from.
    include_solvent : bool, default=False
        Whether to include solvent molecules.

    Returns
    -------
    oechem.OEGraphMol
        Molecule with protein and ligand from `du`
    """
    complex_mol = oechem.OEGraphMol()
    comp_tag = (
        oechem.OEDesignUnitComponents_Protein
        | oechem.OEDesignUnitComponents_Ligand
    )
    if include_solvent:
        comp_tag |= oechem.OEDesignUnitComponents_Solvent
    du.GetComponents(complex_mol, comp_tag)

    return complex_mol


def make_du_from_new_lig(
    initial_complex,
    new_lig,
    dimer=True,
    ref_prot=None,
    split_initial_complex=True,
    split_ref=True,
    ref_chain=None,
    mobile_chain=None,
    loop_db=None,
):
    """
    Create an OEDesignUnit object from the protein component of
    `initial_complex` and the ligand in `new_lig`. Optionally pass in `ref_prot`
    to align the protein part of `initial_complex`.

    Parameters
    ----------
    initial_complex : Union[oechem.OEGraphMol, str]
        Initial complex loaded straight from a PDB file. Can contain ligands,
        waters, cofactors, etc., which will be removed. Can also pass a PDB
        filename instead.
    new_lig : Union[oechem.OEGraphMol, str]
        New ligand molecule (loaded straight from a file). Can also pass a PDB
        or SDF filename instead.
    dimer : bool, default=True
        Whether to build the dimer or just monomer.
    ref_prot : Union[oechem.OEGraphMol, str], optional
        Reference protein to which the protein part of `initial_complex` will
        be aligned. Can also pass a PDB filename instead.
    split_initial_complex : bool, default=True
        Whether to split out protein from `initial_complex`. Setting this to
        False will save time on protein prep if you've already isolated the
        protein.
    split_ref : bool, default=True
        Whether to split out protein from `ref_prot`. Setting this to
        False will save time on protein prep if you've already isolated the
        protein.
    ref_chain : str, optional
        If given, align to given chain in `ref_prot`.
    mobile_chain : str, optional
        If given, align the given chain in the protein component of
        `initial_complex`. If `dimer` is False, this is required and will give
        the chain of the monomer to keep.
    loop_db : str, optional
        File name for the Spruce loop database file.

    Returns
    -------
    oechem.OEDesignUnit
    """

    if (not dimer) and (not mobile_chain):
        raise ValueError(
            "If dimer is False, a value must be given for mobile_chain."
        )

    ## Load initial_complex from file if necessary
    if type(initial_complex) is str:
        initial_complex = load_openeye_pdb(initial_complex, alt_loc=True)
        ## If alt locations are present in PDB file, set positions to highest
        ##  occupancy ALT
        alf = oechem.OEAltLocationFactory(initial_complex)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(initial_complex)

    ## Load ligand from file if necessary
    if type(new_lig) is str:
        if new_lig[-3:] == "sdf":
            parse_func = load_openeye_sdf
        elif new_lig[-3:] == "pdb":
            parse_func = load_openeye_pdb
        else:
            raise ValueError(f"Unknown file format: {new_lig}")

        new_lig = parse_func(new_lig)
    ## Update ligand dimensions in case its internal dimensions property is set
    ##  as 2 for some reason (eg created from SMILES)
    new_lig.SetDimension(3)

    ## Load reference protein from file if necessary
    if type(ref_prot) is str:
        ref_prot = load_openeye_pdb(ref_prot, alt_loc=True)
        ## If alt locations are present in PDB file, set positions to highest
        ##  occupancy ALT
        alf = oechem.OEAltLocationFactory(ref_prot)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(ref_prot)

    ## Split out protein components and align if requested
    if split_initial_complex:
        initial_prot_temp = split_openeye_mol(initial_complex)["pro"]
    else:
        initial_prot_temp = initial_complex

    ## Extract if not dimer
    if dimer:
        initial_prot = initial_prot_temp
    else:
        ### TODO: Have to figure out how to handle water here if it's in a
        ###  different chain from the protein
        initial_prot = oechem.OEGraphMol()
        chain_pred = oechem.OEHasChainID(mobile_chain)
        oechem.OESubsetMol(initial_prot, initial_prot_temp, chain_pred)
    if ref_prot is not None:
        if split_ref:
            ref_prot = split_openeye_mol(ref_prot)["pro"]

        ## Set up predicates
        if ref_chain is not None:
            not_water = oechem.OENotAtom(oechem.OEIsWater())
            ref_chain = oechem.OEHasChainID(ref_chain)
            ref_chain = oechem.OEAndAtom(not_water, ref_chain)
        if mobile_chain is not None:
            try:
                not_water = oechem.OENotAtom(oechem.OEIsWater())
                mobile_chain = oechem.OEHasChainID(mobile_chain)
                mobile_chain = oechem.OEAndAtom(not_water, mobile_chain)
            except Exception as e:
                print(mobile_chain)
                raise e
        initial_prot = superpose_molecule(
            ref_prot, initial_prot, ref_chain, mobile_chain
        )[0]

    ## Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)
    oechem.OEAddExplicitHydrogens(new_lig)

    ## Set up DU building options
    opts = oespruce.OEMakeDesignUnitOptions()
    opts.SetSuperpose(False)
    if loop_db is not None:
        opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

    ## Options set from John's function ########################################
    ## (https://github.com/FoldingAtHome/covid-moonshot/blob/454098f4255467f4655102e0330ebf9da0d09ccb/synthetic-enumeration/sprint-14-quinolones/00-prep-receptor.py)
    opts.GetPrepOptions().SetStrictProtonationMode(True)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment\
    opts.GetSplitOptions().SetMinLigAtoms(5)

    # also consider alternate locations outside binding pocket, important for later filtering
    opts.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(
        False
    )

    # alignment options, only matches are important
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignMethod(
        oechem.OESeqAlignmentMethod_Identity
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignGapPenalty(
        -1
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignExtendPenalty(
        0
    )

    # Both N- and C-termini should be zwitterionic
    # Mpro cleaves its own N- and C-termini
    # See https://www.pnas.org/content/113/46/12997
    opts.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    opts.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
    # Don't allow truncation of termini, since force fields don't have
    #  parameters for this
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetAllowTruncate(
        False
    )
    # Build loops and sidechains
    opts.GetPrepOptions().GetBuildOptions().SetBuildLoops(True)
    opts.GetPrepOptions().GetBuildOptions().SetBuildSidechains(True)

    # Generate ligand tautomers
    opts.GetPrepOptions().GetProtonateOptions().SetGenerateTautomers(True)
    ############################################################################

    ## Finally make new DesignUnit
    du = oechem.OEDesignUnit()
    oespruce.OEMakeDesignUnit(du, initial_prot, new_lig, opts)
    assert du.HasProtein() and du.HasLigand()

    return du


def superpose_molecule(ref_mol, mobile_mol, ref_pred=None, mobile_pred=None):
    """
    Superpose `mobile_mol` onto `ref_mol`.

    Parameters
    ----------
    ref_mol : oechem.OEGraphMol
        Reference molecule to align to.
    mobile_mol : oechem.OEGraphMol
        Molecule to align.
    ref_pred : oechem.OEUnaryPredicate[oechem.OEAtomBase], optional
        Predicate for which atoms to include from `ref_mol`.
    mobile_pred : oechem.OEUnaryPredicate[oechem.OEAtomBase], optional
        Predicate for which atoms to include from `mobile_mol`.

    Returns
    -------
    oechem.OEGraphMol
        New aligned molecule.
    float
        RMSD between `ref_mol` and `mobile_mol` after alignment.
    """

    ## Default atom predicates
    if ref_pred is None:
        ref_pred = oechem.OEIsTrueAtom()
    if mobile_pred is None:
        mobile_pred = oechem.OEIsTrueAtom()

    ## Create object to store results
    aln_res = oespruce.OESuperposeResults()

    ## Set up superposing object and set reference molecule
    superpos = oespruce.OESuperpose()
    superpos.SetupRef(ref_mol, ref_pred)

    ## Perform superposing
    superpos.Superpose(aln_res, mobile_mol, mobile_pred)
    # print(f"RMSD: {aln_res.GetRMSD()}")

    ## Create copy of molecule and transform it to the aligned position
    mobile_mol_aligned = mobile_mol.CreateCopy()
    aln_res.Transform(mobile_mol_aligned)

    return mobile_mol_aligned, aln_res.GetRMSD()


def align_receptor(
    initial_complex,
    dimer=True,
    ref_prot=None,
    split_initial_complex=True,
    split_ref=True,
    ref_chain=None,
    mobile_chain=None,
):
    """
    Function to prepare receptor before building the design unit.

    Returns
    -------

    """
    if (not dimer) and (not mobile_chain):
        raise ValueError(
            "If dimer is False, a value must be given for mobile_chain."
        )

    ## Load initial_complex from file if necessary
    if type(initial_complex) is str:
        initial_complex = load_openeye_pdb(initial_complex, alt_loc=True)
        ## If alt locations are present in PDB file, set positions to highest
        ##  occupancy ALT
        alf = oechem.OEAltLocationFactory(initial_complex)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(initial_complex)

    ## Load reference protein from file if necessary
    if type(ref_prot) is str:
        ref_prot = load_openeye_pdb(ref_prot, alt_loc=True)
        ## If alt locations are present in PDB file, set positions to highest
        ##  occupancy ALT
        alf = oechem.OEAltLocationFactory(ref_prot)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(ref_prot)

    ## Split out protein components and align if requested
    if split_initial_complex:
        initial_prot_temp = split_openeye_mol(initial_complex)["pro"]
    else:
        initial_prot_temp = initial_complex

    ## Extract if not dimer
    if dimer:
        initial_prot = initial_prot_temp
    else:
        ### TODO: Have to figure out how to handle water here if it's in a
        ###  different chain from the protein
        initial_prot = oechem.OEGraphMol()
        chain_pred = oechem.OEHasChainID(mobile_chain)
        oechem.OESubsetMol(initial_prot, initial_prot_temp, chain_pred)
    if ref_prot is not None:
        if split_ref:
            ref_prot = split_openeye_mol(ref_prot)["pro"]

        ## Set up predicates
        if ref_chain is not None:
            not_water = oechem.OENotAtom(oechem.OEIsWater())
            ref_chain = oechem.OEHasChainID(ref_chain)
            ref_chain = oechem.OEAndAtom(not_water, ref_chain)
        if mobile_chain is not None:
            try:
                not_water = oechem.OENotAtom(oechem.OEIsWater())
                mobile_chain = oechem.OEHasChainID(mobile_chain)
                mobile_chain = oechem.OEAndAtom(not_water, mobile_chain)
            except Exception as e:
                print(mobile_chain)
                raise e
        initial_prot = superpose_molecule(
            ref_prot, initial_prot, ref_chain, mobile_chain
        )[0]
    return initial_prot


def mutate_residues(input_mol, res_list, place_h=True):
    """
    Mutate residues in the input molecule using OpenEye.
    TODO: Make this more robust using some kind of sequence alignment.

    Parameters
    ----------
    input_mol : oechem.OEGraphMol
        Input OpenEye molecule
    res_list : List[str]
        List of 3 letter codes for the full sequence of `input_mol`. Must have
        exactly the same number of residues or this will fail
    place_h : bool, default=True
        Whether to place hydrogen atoms

    Returns
    -------
    oechem.OEGraphMol
        Newly mutated molecule
    """
    ## Create a copy of the molecule to avoid modifying original molecule
    mut_prot = input_mol.CreateCopy()
    ## Get sequence of input protein
    input_mol_seq = [r.GetName() for r in oechem.OEGetResidues(input_mol)]
    input_mol_num = [
        r.GetResidueNumber() for r in oechem.OEGetResidues(input_mol)
    ]

    ## Build mutation map from OEResidue to new res name by indexing from res num
    mut_map = {}
    for old_res_name, res_num, r in zip(
        input_mol_seq, input_mol_num, oechem.OEGetResidues(mut_prot)
    ):
        ## Skip if we're looking at a water
        if old_res_name == "HOH":
            continue

        try:
            new_res = res_list[res_num - 1]
        except IndexError:
            ## If the residue number is out of range (because its a water or something weird)
            ## then we can skip right on by it
            continue
        if new_res != old_res_name:
            print(res_num, old_res_name, new_res)
            mut_map[r] = new_res

    ## Mutate and build sidechains
    oespruce.OEMutateResidues(mut_prot, mut_map)

    ## Place hydrogens
    if place_h:
        oechem.OEPlaceHydrogens(mut_prot)

    return mut_prot


def prep_receptor(initial_prot, site_residue="", loop_db=None):
    """
    Prepare DU from protein. If the ligand isn't present in `initial_prot`, a
    value must be provided for `site_residue` or `OEMakeDesignUnits` will fail.

    Parameters
    ----------
    initial_prot : oechem.OEGraphMol
        GraphMol object with protein (and optionally ligand).
    site_residue : str, optional
        Binding site residues, must be of the format “ASP:25: :A”. Optional
        only if `initial_prot` contains a ligand.
    loop_db : str, optional
        Loop database file.

    Returns
    -------
    List[OEDesignUnit]
        Iterator over generated DUs.
    """

    ## Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)

    ## Set up DU building options
    opts = oespruce.OEMakeDesignUnitOptions()
    opts.SetSuperpose(False)
    if loop_db is not None:
        opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

    ## Options set from John's function ########################################
    ## (https://github.com/FoldingAtHome/covid-moonshot/blob/454098f4255467f4655102e0330ebf9da0d09ccb/synthetic-enumeration/sprint-14-quinolones/00-prep-receptor.py)
    opts.GetPrepOptions().SetStrictProtonationMode(True)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment\
    opts.GetSplitOptions().SetMinLigAtoms(5)

    # also consider alternate locations outside binding pocket, important for later filtering
    opts.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(
        False
    )

    # alignment options, only matches are important
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignMethod(
        oechem.OESeqAlignmentMethod_Identity
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignGapPenalty(
        -1
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetSeqAlignExtendPenalty(
        0
    )

    # Both N- and C-termini should be zwitterionic
    # Mpro cleaves its own N- and C-termini
    # See https://www.pnas.org/content/113/46/12997
    opts.GetPrepOptions().GetBuildOptions().SetCapNTermini(False)
    opts.GetPrepOptions().GetBuildOptions().SetCapCTermini(False)
    # Don't allow truncation of termini, since force fields don't have
    #  parameters for this
    opts.GetPrepOptions().GetBuildOptions().GetCapBuilderOptions().SetAllowTruncate(
        False
    )
    # Build loops and sidechains
    opts.GetPrepOptions().GetBuildOptions().SetBuildLoops(True)
    opts.GetPrepOptions().GetBuildOptions().SetBuildSidechains(True)

    # Generate ligand tautomers
    opts.GetPrepOptions().GetProtonateOptions().SetGenerateTautomers(True)
    ############################################################################

    ## Finally make new DesignUnit
    dus = list(
        oespruce.OEMakeDesignUnits(
            initial_prot, oespruce.OEStructureMetadata(), opts, site_residue
        )
    )
    assert dus[0].HasProtein() and dus[0].HasLigand()

    ## Generate docking receptor for each DU
    for du in dus:
        oedocking.OEMakeReceptor(du)

    return dus
