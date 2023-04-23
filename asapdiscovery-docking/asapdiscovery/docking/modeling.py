from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    load_openeye_sdf,
    oechem,
    oespruce,
    split_openeye_mol,
)
from asapdiscovery.modeling.modeling import superpose_molecule


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
        raise ValueError("If dimer is False, a value must be given for mobile_chain.")

    # Load initial_complex from file if necessary
    if type(initial_complex) is str:
        initial_complex = load_openeye_pdb(initial_complex, alt_loc=True)
        # If alt locations are present in PDB file, set positions to highest
        #  occupancy ALT
        alf = oechem.OEAltLocationFactory(initial_complex)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(initial_complex)

    # Load ligand from file if necessary
    if type(new_lig) is str:
        if new_lig[-3:] == "sdf":
            parse_func = load_openeye_sdf
        elif new_lig[-3:] == "pdb":
            parse_func = load_openeye_pdb
        else:
            raise ValueError(f"Unknown file format: {new_lig}")

        new_lig = parse_func(new_lig)
    # Update ligand dimensions in case its internal dimensions property is set
    #  as 2 for some reason (eg created from SMILES)
    new_lig.SetDimension(3)

    # Load reference protein from file if necessary
    if type(ref_prot) is str:
        ref_prot = load_openeye_pdb(ref_prot, alt_loc=True)
        # If alt locations are present in PDB file, set positions to highest
        #  occupancy ALT
        alf = oechem.OEAltLocationFactory(ref_prot)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(ref_prot)

    # Split out protein components and align if requested
    if split_initial_complex:
        initial_prot_temp = split_openeye_mol(initial_complex)["pro"]
    else:
        initial_prot_temp = initial_complex

    # Extract if not dimer
    if dimer:
        initial_prot = initial_prot_temp
    else:
        # TODO: Have to figure out how to handle water here if it's in a
        #  different chain from the protein
        initial_prot = oechem.OEGraphMol()
        chain_pred = oechem.OEHasChainID(mobile_chain)
        oechem.OESubsetMol(initial_prot, initial_prot_temp, chain_pred)
    if ref_prot is not None:
        if split_ref:
            ref_prot = split_openeye_mol(ref_prot)["pro"]

        # Set up predicates
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

    # Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)
    oechem.OEAddExplicitHydrogens(new_lig)

    # Set up DU building options
    opts = oespruce.OEMakeDesignUnitOptions()
    opts.SetSuperpose(False)
    if loop_db is not None:
        opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
            loop_db
        )

    # Options set from John's function ####################
    # (https://github.com/FoldingAtHome/covid-moonshot/blob/454098f4255467f4655102e0330ebf9da0d09ccb/synthetic-enumeration/sprint-14-quinolones/00-prep-receptor.py)
    opts.GetPrepOptions().SetStrictProtonationMode(True)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment\
    opts.GetSplitOptions().SetMinLigAtoms(5)

    # also consider alternate locations outside binding pocket, important for later
    # filtering
    opts.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(False)

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
    ######################################

    # Finally make new DesignUnit
    du = oechem.OEDesignUnit()
    oespruce.OEMakeDesignUnit(du, initial_prot, new_lig, opts)
    assert du.HasProtein() and du.HasLigand()

    return du
