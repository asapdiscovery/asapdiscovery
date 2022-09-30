from openeye import oechem, oedocking, oespruce

from .datasets.utils import (
    load_openeye_pdb,
    load_openeye_sdf,
    split_openeye_mol,
)


def du_to_complex(du):
    """
    Convert OEDesignUnit to OEGraphMol containing the protein and ligand from
    `du`.

    Parameters
    ----------
    du : oechem.OEDesignUnit
        OEDesignUnit object to extract from.

    Returns
    -------
    oechem.OEGraphMol
        Molecule with protein and ligand from `du`
    """
    complex_mol = oechem.OEGraphMol()
    du.GetComponents(
        complex_mol,
        oechem.OEDesignUnitComponents_Protein
        | oechem.OEDesignUnitComponents_Ligand,
    )

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
    input_prot,
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
    input_prot
    if type(input_prot) is str:
        initial_complex = load_openeye_pdb(input_prot, alt_loc=True)
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


def mutate_residues(input_mol, res_list):
    ## Try using direct mutation instead
    hierview = oechem.OEHierView(input_mol)
    residues = [residue.GetOEResidue() for residue in hierview.GetResidues()]
    for residue in residues:
        res_num = residue.GetResidueNumber()
        res_name = residue.GetName()
        if res_num < len(res_list):
            try:
                desired_res = res_list[res_num - 1]
                if res_name != desired_res and len(res_name) > 0:
                    print(res_name)
                    print(res_num)
                    print(desired_res)
                    # print(
                    #     f"Mutating {str(res_name)}{res_num} to {desired_res} in chain {residue.GetChainID()}"
                    # )
                    if not oespruce.OEMutateResidue(
                        input_mol, residue, desired_res
                    ):
                        print("Failed")
            except IndexError:
                print(
                    "skipping since residue number was out of range for the sequence"
                )
                pass

    return input_mol


def prep_receptor(
    initial_prot,
    site_residue,
    sequence=None,
    loop_db=None,
):
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
    # TODO: Add ability to add SEQRES and mutate protein accordingly

    ## Finally make new DesignUnit
    ## Using this function instead of OEMakeDesignUnit enables passing the empty 'metadata'
    ## object which makes it possible to build an empty DU
    metadata = oespruce.OEStructureMetadata()
    if sequence:
        # Use Sequence Metadata Class to add sequence
        seq_meta = oespruce.OESequenceMetadata()
        seq_meta.SetSequence(sequence)
        metadata.AddSequenceMetadata(seq_meta)
        print(metadata.GetSequenceMetadata()[0].GetSequence())

    print(result)
    # print("Making DU")
    # design_units = oespruce.OEMakeDesignUnits(
    #     initial_prot, metadata, opts, site_residue
    # )
    # oespruce.OESpruceFilter(du, initial_prot, opts)
    # assert du.HasProtein()
    # print(design_units)
    #
    # return design_units
    return initial_prot
