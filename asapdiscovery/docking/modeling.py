import datetime
import logging
import os

from openeye import oechem, oedocking, oespruce

from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    load_openeye_sdf,
    split_openeye_mol,
    openeye_perceive_residues,
    save_openeye_pdb,
)
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.utils import seqres_to_res_list


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
    keep_water=True,
):
    """
    Basically a copy of the above function to generate an aligned receptor without also needing to do the rest of the
    protein prep.

    Parameters
    ----------
    initial_complex : Union[oechem.OEGraphMol, str]
        Initial complex loaded straight from a PDB file. Can contain ligands,
        waters, cofactors, etc., which will be removed. Can also pass a PDB
        filename instead.
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
    keep_water : bool, default=True
        Whether or not to keep the crystallographic water molecules.

    Returns
    -------
    oechem.OEDesignUnit

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
        split_dict = split_openeye_mol(initial_complex)
        initial_prot_temp = split_dict["pro"]
        if keep_water:
            oechem.OEAddMols(initial_prot_temp, split_dict["water"])
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


def mutate_residues(input_mol, res_list, protein_chains=None, place_h=True):
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
    input_mol_chain = [
        r.GetExtChainID() for r in oechem.OEGetResidues(input_mol)
    ]
    input_mol_seq = [r.GetName() for r in oechem.OEGetResidues(input_mol)]
    input_mol_num = [
        r.GetResidueNumber() for r in oechem.OEGetResidues(input_mol)
    ]

    ## Build mutation map from OEResidue to new res name by indexing from res num
    mut_map = {}
    for old_res_name, res_num, chain, r in zip(
        input_mol_seq,
        input_mol_num,
        input_mol_chain,
        oechem.OEGetResidues(mut_prot),
    ):
        ## Skip if not in identified protein chains
        if protein_chains:
            if chain not in protein_chains:
                continue
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

    ## Return early if no mutations found
    if len(mut_map) == 0:
        print("No mutations found", flush=True)
        return input_mol

    ## Mutate and build sidechains
    oespruce.OEMutateResidues(mut_prot, mut_map)

    ## Place hydrogens
    if place_h:
        oechem.OEPlaceHydrogens(mut_prot)

    ## Re-percieve residues so that atom number and connect records dont get screwed up
    openeye_perceive_residues(mut_prot)

    return mut_prot


def prep_receptor(
    initial_prot, site_residue="", loop_db=None, protein_only=False, seqres=None
):
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
    protein_only : bool, default=True
        Whether we want to only keep the protein or keep the ligand as well.
    seqres : str, optional
        Residue sequence of a single chain. Should be in the format of three
        letter codes, separated by a space for each residue.

    Returns
    -------
    List[OEDesignUnit]
        Iterator over generated DUs.
    """
    # initial_prot = build_dimer_from_monomer(initial_prot)

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

    if not protein_only:
        # Generate ligand tautomers
        opts.GetPrepOptions().GetProtonateOptions().SetGenerateTautomers(True)

    ############################################################################

    ## Structure metadata object
    metadata = oespruce.OEStructureMetadata()

    ## Allow for adding residues at the beginning/end if they're missing
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetBuildTails(
        True
    )
    if seqres:
        all_prot_chains = {
            res.GetExtChainID()
            for res in oechem.OEGetResidues(initial_prot)
            if (res.GetName() != "LIG") and (res.GetName() != "HOH")
        }
        for chain in all_prot_chains:
            seq_metadata = oespruce.OESequenceMetadata()
            seq_metadata.SetChainID(chain)
            seq_metadata.SetSequence(seqres)
            metadata.AddSequenceMetadata(seq_metadata)

    ## Finally make new DesignUnit
    dus = list(
        oespruce.OEMakeDesignUnits(initial_prot, metadata, opts, site_residue)
    )
    assert dus[0].HasProtein()
    if not protein_only:
        assert dus[0].HasLigand()

    ## Generate docking receptor for each DU
    for du in dus:
        oedocking.OEMakeReceptor(du)

    return dus


def build_dimer_from_monomer(prot):
    ## Build monomer into dimer as necessary (will need to handle
    ##  re-labeling chains since the monomer seems to get the chainID C)
    ## Shouldn't affect the protein if the dimer has already been built
    bus = list(oespruce.OEExtractBioUnits(prot))

    ## Need to cast to OEGraphMol bc returned type is OEMolBase, which
    ##  doesn't pickle
    prot = oechem.OEGraphMol(bus[0])

    ## Keep track of chain IDs
    all_chain_ids = {
        r.GetExtChainID()
        for r in oechem.OEGetResidues(prot)
        if all(
            [
                not oechem.OEIsWater()(a)
                for a in oechem.OEGetResidueAtoms(prot, r)
            ]
        )
    }
    if len(all_chain_ids) != 2:
        raise AssertionError(f"Chains: {all_chain_ids}")

    print(all_chain_ids)
    return prot


def remove_extra_ligands(mol, lig_chain=None):
    """
    Remove extra ligands from a molecule. Useful in the case where a complex
    crystal structure has two copies of the ligand, but we only want one. If
    `lig_chain` is not specified, we'll automatically select the first chain
    (as sorted alphabetically) to be the copy to keep.

    Creates a copy of the input molecule, so input molecule is not modified.

    Parameters
    ----------
    mol : oechem.OEMolBase
        Complex molecule.
    lig_chain : str, optional
        Ligand chain ID to keep.

    Returns
    -------
    oechem.OEMolBase
        Molecule with extra ligand copies removed.
    """
    ## Atom filter to match all atoms in residue with name LIG
    all_lig_match = oechem.OEAtomMatchResidueID()
    all_lig_match.SetName("LIG")
    all_lig_filter = oechem.OEAtomMatchResidue(all_lig_match)

    ## Detect ligand chain to keep if none is given
    if lig_chain is None:
        lig_chain = sorted(
            {
                oechem.OEAtomGetResidue(a).GetExtChainID()
                for a in mol.GetAtoms(all_lig_filter)
            }
        )[0]

    ## Copy molecule and delete all lig atoms that don't have the desired chain
    mol_copy = mol.CreateCopy()
    for a in mol_copy.GetAtoms(all_lig_filter):
        if oechem.OEAtomGetResidue(a).GetExtChainID() != lig_chain:
            mol_copy.DeleteAtom(a)
    return mol_copy


def check_completed(d, prefix):
    """
    Check if this prep process has already been run successfully in the given
    directory.

    Parameters
    ----------
    d : str
        Directory to check.

    Returns
    -------
    bool
        True if both files exist and can be loaded, otherwise False.
    """

    if (
        not os.path.isfile(os.path.join(d, f"{prefix}_prepped_receptor_0.oedu"))
    ) or (
        not os.path.isfile(os.path.join(d, f"{prefix}_prepped_receptor_0.pdb"))
    ):
        return False

    try:
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(
            os.path.join(d, f"{prefix}_prepped_receptor_0.oedu"), du
        )
    except Exception:
        return False

    try:
        _ = load_openeye_pdb(
            os.path.join(d, f"{prefix}_prepped_receptor_0.pdb")
        )
    except Exception:
        return False

    return True


def prep_mp(
    xtal: CrystalCompoundData,
    ref_prot,
    seqres,
    out_base,
    loop_db,
    protein_only: bool,
):
    ## Make output directory
    out_dir = os.path.join(out_base, f"{xtal.output_name}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## Prepare logger
    handler = logging.FileHandler(os.path.join(out_dir, f"{xtal.output_name}-log.txt"), mode="w")
    prep_logger = logging.getLogger(xtal.output_name)
    prep_logger.setLevel(logging.INFO)
    prep_logger.addHandler(handler)
    prep_logger.info(datetime.datetime.isoformat(datetime.datetime.now()))

    ## Check if results already exist
    if check_completed(out_dir, xtal.output_name):
        prep_logger.info("Already completed! Finishing.")
        return
    prep_logger.info(f"Prepping {xtal.output_name}")

    ## Load protein from pdb
    initial_prot = load_openeye_pdb(xtal.str_fn)

    if seqres:
        res_list = seqres_to_res_list(seqres)
        prep_logger.info("Mutating to provided seqres")

        ## Mutate the residues to match the residue list
        initial_prot = mutate_residues(
            initial_prot, res_list, xtal.protein_chains
        )

    ## Delete extra copies of ligand in the complex
    initial_prot = remove_extra_ligands(
        initial_prot, lig_chain=xtal.active_site_chain
    )

    if ref_prot:
        prep_logger.info("Aligning receptor")
        initial_prot = align_receptor(
            initial_complex=initial_prot,
            ref_prot=ref_prot,
            dimer=True,
            split_initial_complex=protein_only,
            mobile_chain=xtal.active_site_chain,
            ref_chain="A",
        )
        # prone to race condition if multiple processes are writing to same file
        # so need a file prefix 
        save_openeye_pdb(initial_prot, f"{xtal.output_name}-align_test.pdb")
    ## Take the first returned DU and save it
    try:
        prep_logger.info("Attempting to prepare design units")
        site_residue = xtal.active_site if xtal.active_site else ""
        design_units = prep_receptor(
            initial_prot,
            site_residue=site_residue,
            loop_db=loop_db,
            protein_only=protein_only,
            seqres=" ".join(res_list),
        )
    except IndexError as e:
        prep_logger.error(
            f"DU generation failed for {xtal.output_name}",
        )
        return

    du = design_units[0]
    for i, du in enumerate(design_units):
        success = oechem.OEWriteDesignUnit(
            os.path.join(
                out_dir, f"{xtal.output_name}_prepped_receptor_{i}.oedu"
            ),
            du,
        )
        prep_logger.info(
            f"{xtal.output_name} DU successfully written out: {success}"
        )

        ## Save complex as PDB file
        complex_mol = du_to_complex(du, include_solvent=True)

        ## TODO: Compare this function to Ben's code below
        # openeye_copy_pdb_data(complex_mol, initial_prot, "SEQRES")

        ## Add SEQRES entries if they're not present
        if (not oechem.OEHasPDBData(complex_mol, "SEQRES")) and seqres:
            for seqres_line in seqres.split("\n"):
                if seqres_line != "":
                    oechem.OEAddPDBData(complex_mol, "SEQRES", seqres_line[6:])

        save_openeye_pdb(
            complex_mol,
            os.path.join(
                out_dir, f"{xtal.output_name}_prepped_receptor_{i}.pdb"
            ),
        )

    prep_logger.info(
        f"Finished protein prep at {datetime.datetime.isoformat(datetime.datetime.now())}"
    )
