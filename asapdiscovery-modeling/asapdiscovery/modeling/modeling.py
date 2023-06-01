import datetime
import logging
import os
from pathlib import Path
from asapdiscovery.modeling.schema import MoleculeFilter

from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    oechem,
    oedocking,
    oespruce,
    openeye_perceive_residues,
    save_openeye_pdb,
)
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.utils import seqres_to_res_list
from collections import namedtuple


def add_seqres_to_openeye_protein(
    prot: oechem.OEGraphMol, seqres: str = None
) -> oechem.OEGraphMol:
    """
    Adds the SEQRES metadata to the given protein structure.

    Args:
    - prot (oechem.OEDesignUnit): the protein structure to add the SEQRES metadata to.
    """
    # Add SEQRES entries if they're not present
    if (not oechem.OEHasPDBData(prot, "SEQRES")) and seqres:
        for seqres_line in seqres.split("\n"):
            if seqres_line != "":
                oechem.OEAddPDBData(prot, "SEQRES", seqres_line[6:])
    return prot


def spruce_protein(
    initial_prot: oechem.OEGraphMol,
    return_du=False,
    seqres: str = None,
    loop_db: Path = None,
    site_residue=None,  # e.g. "HIS:41: :A:0: "
) -> oechem.OEDesignUnit or oechem.OEGraphMol:
    """
    Applies the OESpruce protein preparation pipeline to the given protein structure.

    Args:
    - initial_prot (oechem.OEMol): the input protein structure to be prepared.
    - return_du (bool, optional): whether to return a design unit (DU) as the output. If True, the function will
      return the DU generated by the OESpruce pipeline; if False, the function will return the prepared protein
      structure as an oechem.OEGraphMol. Default is False.
    - seqres (str, optional): the SEQRES string of the protein. If provided, the SEQRES metadata will be added
      to the structure before applying the OESpruce pipeline. Default is None.
    - loop_db (str, optional): the filename of the loop database to be used by the OESpruce pipeline. If provided,
      the pipeline will include loop building step. Default is None.
    - site_residue (str, optional): the site residue used to define the binding site of the protein. Default is
      "HIS:41: :A:0: ". This is necessary when there is no ligand in the input structure, otherwise OpenEye will not
      know where to put the binding site.

    Returns:
    - oechem.OEMol: the prepared protein structure, or a DU generated by the OESpruce pipeline if return_du is True.
      If design unit preparation fails, it will return the prepared protein structure instead.
    """
    from asapdiscovery.data.openeye import openeye_perceive_residues

    # Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)

    # Set up DU building options
    opts = oespruce.OEMakeDesignUnitOptions()
    opts.SetSuperpose(False)
    # Options set from John's function ########################################
    # (https://github.com/FoldingAtHome/covid-moonshot/blob/454098f4255467f4655102e0330ebf9da0d09ccb/synthetic-enumeration/sprint-14-quinolones/00-prep-receptor.py)
    opts.GetPrepOptions().SetStrictProtonationMode(True)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment\
    opts.GetSplitOptions().SetMinLigAtoms(5)

    # also consider alternate locations outside binding pocket, important for later filtering
    opts.GetPrepOptions().GetEnumerateSitesOptions().SetCollapseNonSiteAlts(True)

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
    # Set Build Loop and Sidechain Opts
    sc_opts = oespruce.OESidechainBuilderOptions()

    loop_opts = oespruce.OELoopBuilderOptions()
    loop_opts.SetSeqAlignMethod(oechem.OESeqAlignmentMethod_Identity)
    loop_opts.SetSeqAlignGapPenalty(-1)
    loop_opts.SetSeqAlignExtendPenalty(0)

    # Allow for adding residues at the beginning/end if they're missing
    loop_opts.SetBuildTails(True)

    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetBuildTails(True)

    if loop_db is not None:
        print("Adding loop db")
        loop_opts.SetLoopDBFilename(str(loop_db))

    # Structure metadata object
    metadata = oespruce.OEStructureMetadata()

    # Add SEQRES metadata
    if seqres:
        print("adding seqres")
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

    # Construct spruce filter
    spruce_opts = oespruce.OESpruceFilterOptions()
    spruce = oespruce.OESpruceFilter(spruce_opts, opts)

    # Spruce!
    from asapdiscovery.data.openeye import oegrid

    # This object is for some reason needed in order to run spruce
    grid = oegrid.OESkewGrid()

    oespruce.OEBuildLoops(initial_prot, metadata, sc_opts, loop_opts)
    oespruce.OEBuildSidechains(initial_prot, sc_opts)
    oechem.OEPlaceHydrogens(initial_prot)
    spruce.StandardizeAndFilter(initial_prot, grid, metadata)

    # Re-percieve residues so that atom number and connect records dont get screwed up
    openeye_perceive_residues(initial_prot)

    # If we don't want to return a DU, just return the spruced protein
    if not return_du:
        return initial_prot

    if site_residue:
        dus = list(
            oespruce.OEMakeDesignUnits(initial_prot, metadata, opts, site_residue)
        )
    else:
        dus = list(oespruce.OEMakeDesignUnits(initial_prot, metadata, opts))
    try:
        du = dus[0]
        if not du.HasProtein():
            raise ValueError(f"Resulting design unit '{du.GetTitle()}' has no protein.")
        if not site_residue and not du.HasLigand():
            raise ValueError(f"Resulting design unit '{du.GetTitle()}' has no ligand.")
        # Generate docking receptor for each DU
        oedocking.OEMakeReceptor(du)

        return du

    except IndexError:
        return initial_prot


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

    # Default atom predicates
    if ref_pred is None:
        ref_pred = oechem.OEIsTrueAtom()
    if mobile_pred is None:
        mobile_pred = oechem.OEIsTrueAtom()

    # Create object to store results
    aln_res = oespruce.OESuperposeResults()

    # Set up superposing object and set reference molecule
    superpos = oespruce.OESuperpose()
    superpos.SetupRef(ref_mol, ref_pred)

    # Perform superposing
    superpos.Superpose(aln_res, mobile_mol, mobile_pred)
    # print(f"RMSD: {aln_res.GetRMSD()}")

    # Create copy of molecule and transform it to the aligned position
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
    Basically a copy of the above function to generate an aligned receptor without also
    needing to do the rest of the protein prep.

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
        raise ValueError("If dimer is False, a value must be given for mobile_chain.")

    # Load initial_complex from file if necessary
    if type(initial_complex) is str:
        initial_complex = load_openeye_pdb(initial_complex, alt_loc=True)
        # If alt locations are present in PDB file, set positions to highest
        #  occupancy ALT
        alf = oechem.OEAltLocationFactory(initial_complex)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(initial_complex)

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
        split_dict = split_openeye_mol(initial_complex)
        initial_prot_temp = split_dict["pro"]
        if keep_water:
            oechem.OEAddMols(initial_prot_temp, split_dict["water"])
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
    # Create a copy of the molecule to avoid modifying original molecule
    mut_prot = input_mol.CreateCopy()
    # Get sequence of input protein
    input_mol_chain = [r.GetExtChainID() for r in oechem.OEGetResidues(input_mol)]
    input_mol_seq = [r.GetName() for r in oechem.OEGetResidues(input_mol)]
    input_mol_num = [r.GetResidueNumber() for r in oechem.OEGetResidues(input_mol)]

    # Build mutation map from OEResidue to new res name by indexing from res num
    mut_map = {}
    for old_res_name, res_num, chain, r in zip(
        input_mol_seq,
        input_mol_num,
        input_mol_chain,
        oechem.OEGetResidues(mut_prot),
    ):
        # Skip if not in identified protein chains
        if protein_chains:
            if chain not in protein_chains:
                continue
        # Skip if we're looking at a water
        if old_res_name == "HOH":
            continue
        try:
            new_res = res_list[res_num - 1]
        except IndexError:
            # If the residue number is out of range (because its a water or something
            # weird) then we can skip right on by it
            continue
        if new_res != old_res_name:
            print(res_num, old_res_name, new_res)
            mut_map[r] = new_res

    # Return early if no mutations found
    if len(mut_map) == 0:
        print("No mutations found", flush=True)
        return input_mol

    # Mutate and build sidechains
    oespruce.OEMutateResidues(mut_prot, mut_map)

    # Place hydrogens
    if place_h:
        oechem.OEPlaceHydrogens(mut_prot)

    # Re-percieve residues so that atom number and connect records dont get screwed up
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

    # Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)

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

    if not protein_only:
        # Generate ligand tautomers
        opts.GetPrepOptions().GetProtonateOptions().SetGenerateTautomers(True)

    ######################################

    # Structure metadata object
    metadata = oespruce.OEStructureMetadata()

    # Allow for adding residues at the beginning/end if they're missing
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetBuildTails(True)
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

    # Finally make new DesignUnit
    dus = list(oespruce.OEMakeDesignUnits(initial_prot, metadata, opts, site_residue))
    assert dus[0].HasProtein()
    if not protein_only:
        assert dus[0].HasLigand()

    # Generate docking receptor for each DU
    for du in dus:
        oedocking.OEMakeReceptor(du)

    return dus


def build_dimer_from_monomer(prot):
    # Build monomer into dimer as necessary (will need to handle
    #  re-labeling chains since the monomer seems to get the chainID C)
    # Shouldn't affect the protein if the dimer has already been built
    bus = list(oespruce.OEExtractBioUnits(prot))

    # Need to cast to OEGraphMol bc returned type is OEMolBase, which
    #  doesn't pickle
    prot = oechem.OEGraphMol(bus[0])

    # Keep track of chain IDs
    all_chain_ids = {
        r.GetExtChainID()
        for r in oechem.OEGetResidues(prot)
        if all([not oechem.OEIsWater()(a) for a in oechem.OEGetResidueAtoms(prot, r)])
    }
    if len(all_chain_ids) != 2:
        raise AssertionError(f"Chains: {all_chain_ids}")

    print(all_chain_ids)
    return prot


def find_ligand_chains(mol: oechem.OEMolBase):
    """
    Find the chains in a molecule that contain the ligand, identified by the resid "LIG". This is useful for
    cases where the ligand is present in multiple chains.

    Parameters
    ----------
    mol : oechem.OEMolBase
        Complex molecule.

    Returns
    -------
    List[str]
        List of chain IDs that contain the ligand.
    """
    lig_chain_ids = set()
    for res in oechem.OEGetResidues(mol):
        if res.GetName() == "LIG":
            lig_chain_ids.add(res.GetChainID())
    return list(sorted(lig_chain_ids))


def find_protein_chains(mol: oechem.OEMolBase):
    """
    Find the chains in a molecule that contain the protein.

    Parameters
    ----------
    mol : oechem.OEMolBase
        Complex molecule.

    Returns
    -------
    List[str]
        List of chain IDs that contain the protein.
    """
    prot_chain_ids = set()
    for res in oechem.OEGetResidues(mol):
        if oechem.OEIsStandardProteinResidue(res):
            prot_chain_ids.add(res.GetChainID())
    return list(sorted(prot_chain_ids))


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
    # Atom filter to match all atoms in residue with name LIG
    all_lig_match = oechem.OEAtomMatchResidueID()
    all_lig_match.SetName("LIG")
    all_lig_filter = oechem.OEAtomMatchResidue(all_lig_match)

    lig_chains = find_ligand_chains(mol)
    # Detect ligand chain to keep if none is given
    if lig_chain is None:
        lig_chain = lig_chains[0]

    # Copy molecule and delete all lig atoms that don't have the desired chain
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

    if (not os.path.isfile(os.path.join(d, f"{prefix}_prepped_receptor_0.oedu"))) or (
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
        _ = load_openeye_pdb(os.path.join(d, f"{prefix}_prepped_receptor_0.pdb"))
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
    """
    Prepare a crystal structure for docking simulations.

    The function pre-processes a crystal structure file in PDB format, prepares a design unit (DU)
    for docking simulations, and saves the pre-processed structures and DUs in the specified output
    directory. The pre-processing steps include:
    - Creating the output directory if it does not exist
    - Setting up a logger for tracking progress and errors
    - Checking if the output files already exist, and exiting early if they do
    - Loading the protein from the input PDB file
    - Mutating the residues of the protein to match the provided SEQRES sequence (if any)
    - Removing extra copies of the ligand in the protein-ligand complex
    - Aligning the protein to a reference protein (if provided)
    - Preparing a DU for docking simulations using the initial protein structure and loop database
    - Saving the DU and pre-processed structures in the output directory

    Args:
        xtal (CrystalCompoundData): An object containing information about the crystal structure.
        ref_prot (Optional): A reference protein structure to align the input protein to.
        seqres (Optional): A string containing the SEQRES sequence to mutate the protein to.
        out_base (str): The base output directory for the pre-processed files.
        loop_db (str): The path to the loop database file.
        protein_only (bool): Whether to include only the protein in the DU, or also include the ligand.

    Returns:
        None: The function writes the pre-processed files to disk but does not return any value.
    """
    # Make output directory
    out_dir = os.path.join(out_base, f"{xtal.output_name}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Prepare logger
    handler = logging.FileHandler(
        os.path.join(out_dir, f"{xtal.output_name}-log.txt"), mode="w"
    )
    prep_logger = logging.getLogger(xtal.output_name)
    prep_logger.setLevel(logging.INFO)
    prep_logger.addHandler(handler)
    prep_logger.info(datetime.datetime.isoformat(datetime.datetime.now()))

    # Check if results already exist
    if check_completed(out_dir, xtal.output_name):
        prep_logger.info("Already completed! Finishing.")
        return
    prep_logger.info(f"Prepping {xtal.output_name}")

    # Load protein from pdb
    initial_prot = load_openeye_pdb(xtal.str_fn)

    if seqres:
        res_list = seqres_to_res_list(seqres)
        prep_logger.info("Mutating to provided seqres")

        # Mutate the residues to match the residue list
        initial_prot = mutate_residues(initial_prot, res_list, xtal.protein_chains)

        # Build seqres here
        seqres = " ".join(res_list)

    # Delete extra copies of ligand in the complex
    initial_prot = remove_extra_ligands(initial_prot, lig_chain=xtal.active_site_chain)

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
    # Take the first returned DU and save it
    try:
        prep_logger.info("Attempting to prepare design units")
        site_residue = xtal.active_site if xtal.active_site else ""
        design_units = prep_receptor(
            initial_prot,
            site_residue=site_residue,
            loop_db=loop_db,
            protein_only=protein_only,
            seqres=seqres,
        )
    except IndexError as e:
        prep_logger.error(
            f"DU generation failed for {xtal.output_name} with error {str(e)}",
        )
        return

    du = design_units[0]
    for i, du in enumerate(design_units):
        success = oechem.OEWriteDesignUnit(
            os.path.join(out_dir, f"{xtal.output_name}_prepped_receptor_{i}.oedu"),
            du,
        )
        prep_logger.info(f"{xtal.output_name} DU successfully written out: {success}")

        # Save complex as PDB file
        complex_mol = du_to_complex(du, include_solvent=True)

        # TODO: Compare this function to Ben's code below
        # openeye_copy_pdb_data(complex_mol, initial_prot, "SEQRES")

        # Add SEQRES entries if they're not present
        if (not oechem.OEHasPDBData(complex_mol, "SEQRES")) and seqres:
            for seqres_line in seqres.split("\n"):
                if seqres_line != "":
                    oechem.OEAddPDBData(complex_mol, "SEQRES", seqres_line[6:])

        save_openeye_pdb(
            complex_mol,
            os.path.join(out_dir, f"{xtal.output_name}_prepped_receptor_{i}.pdb"),
        )

    prep_logger.info(
        f"Finished protein prep at {datetime.datetime.isoformat(datetime.datetime.now())}"
    )


def split_openeye_mol_alt(complex_mol, molecule_filter: MoleculeFilter) -> namedtuple:
    """
    Split an OpenEye-loaded molecule into protein, ligand, etc.
    Uses the OpenEye OESplitMolComplex function, which automatically splits out
    only the first ligand binding site it sees.

    Parameters
    ----------
    complex_mol : oechem.OEMolBase
        Complex molecule to split.
    molecule_filter : MoleculeFilter
        Molecule filter object that contains the filter criteria.

    Returns
    -------
    namedtuple
        A namedtuple containing the protein, ligand, and water molecules.
    """
    # These are the objects that will be returned at the end
    lig_mol = oechem.OEGraphMol()
    prot_mol = oechem.OEGraphMol()
    water_mol = oechem.OEGraphMol()
    oth_mol = oechem.OEGraphMol()

    # Make splitting split out covalent ligands possible
    # TODO: look into different covalent-related options here
    opts = oechem.OESplitMolComplexOptions()
    opts.SetSplitCovalent(True)
    opts.SetSplitCovalentCofactors(True)

    # Protein splitting options
    complex_filter = []
    if "protein" in molecule_filter.components_to_keep:
        prot_only = oechem.OEMolComplexFilterFactory(
            oechem.OEMolComplexFilterCategory_Protein
        )
        if len(molecule_filter.protein_chains) > 0:
            chain_filters = [
                oechem.OERoleMolComplexFilterFactory(
                    oechem.OEMolComplexChainRoleFactory(chain)
                )
                for chain in molecule_filter.protein_chains
            ]
            if len(chain_filters) > 1:
                chain_filter = oechem.OEOrRoleSet(*chain_filters)
            else:
                chain_filter = chain_filters[0]
            prot_filter = oechem.OEAndRoleSet(prot_only, chain_filter)
        else:
            prot_filter = prot_only
        opts.SetProteinFilter(prot_filter)
        complex_filter.append(prot_filter)
    # Ligand splitting options
    # Select ligand as all residues with resn LIG
    if "ligand" in molecule_filter.components_to_keep:
        lig_only = oechem.OEMolComplexFilterFactory(
            oechem.OEMolComplexFilterCategory_Ligand
        )
        if molecule_filter.ligand_chain is None:
            lig_filter = lig_only
        else:
            lig_chain = oechem.OERoleMolComplexFilterFactory(
                oechem.OEMolComplexChainRoleFactory(molecule_filter.ligand_chain)
            )
            lig_filter = oechem.OEAndRoleSet(lig_only, lig_chain)
        opts.SetLigandFilter(lig_filter)
        complex_filter.append(lig_filter)

    # If only one argument is passed to OEOrRoleSet, it will throw an error, annoyingly
    if len(complex_filter) == 0:
        raise RuntimeError(
            "No components to keep were specified. Maybe OpenEye wackiness?"
        )
    if len(complex_filter) == 1:
        opts.SetProteinFilter(*complex_filter)

    else:
        opts.SetProteinFilter(oechem.OEOrRoleSet(*complex_filter))
    oechem.OESplitMolComplex(
        lig_mol,
        prot_mol,
        water_mol,
        oth_mol,
        complex_mol,
        opts,
    )

    # split_mol = namedtuple(
    #     "MoleculeSplit", ["protein", "ligand", "water", "other"]
    # )
    # return split_mol(prot_mol, lig_mol, water_mol, oth_mol)
    return prot_mol


def split_openeye_mol(complex_mol, lig_chain="A", prot_cutoff_len=10):
    """
    Split an OpenEye-loaded molecule into protein, ligand, etc.
    Uses the OpenEye OESplitMolComplex function, which automatically splits out
    only the first ligand binding site it sees.

    Parameters
    ----------
    complex_mol : oechem.OEMolBase
        Complex molecule to split.
    lig_chain : str, default="A"
        Which copy of the ligand to keep. Pass None to keep all ligand atoms.
    prot_cutoff_len : int, default=10
        Minimum number of residues in a protein chain required in order to keep

    Returns
    -------
    """

    # Test splitting
    lig_mol = oechem.OEGraphMol()
    prot_mol = oechem.OEGraphMol()
    water_mol = oechem.OEGraphMol()
    oth_mol = oechem.OEGraphMol()

    # Make splitting split out covalent ligands
    # TODO: look into different covalent-related options here
    opts = oechem.OESplitMolComplexOptions()
    opts.SetSplitCovalent(True)
    opts.SetSplitCovalentCofactors(True)

    # Select protein as all protein atoms in chain A or chain B
    prot_only = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_Protein
    )
    a_chain = oechem.OERoleMolComplexFilterFactory(
        oechem.OEMolComplexChainRoleFactory("A")
    )
    b_chain = oechem.OERoleMolComplexFilterFactory(
        oechem.OEMolComplexChainRoleFactory("B")
    )
    a_or_b_chain = oechem.OEOrRoleSet(a_chain, b_chain)
    opts.SetProteinFilter(oechem.OEAndRoleSet(prot_only, a_or_b_chain))

    # Select ligand as all residues with resn LIG
    lig_only = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_Ligand
    )
    if lig_chain is None:
        opts.SetLigandFilter(lig_only)
    else:
        lig_chain = oechem.OERoleMolComplexFilterFactory(
            oechem.OEMolComplexChainRoleFactory(lig_chain)
        )
        opts.SetLigandFilter(oechem.OEAndRoleSet(lig_only, lig_chain))

    # Set water filter (keep all waters in A, B, or W chains)
    #  (is this sufficient? are there other common water chain ids?)
    wat_only = oechem.OEMolComplexFilterFactory(oechem.OEMolComplexFilterCategory_Water)
    w_chain = oechem.OERoleMolComplexFilterFactory(
        oechem.OEMolComplexChainRoleFactory("W")
    )
    all_wat_chains = oechem.OEOrRoleSet(a_or_b_chain, w_chain)
    opts.SetWaterFilter(oechem.OEAndRoleSet(wat_only, all_wat_chains))

    oechem.OESplitMolComplex(
        lig_mol,
        prot_mol,
        water_mol,
        oth_mol,
        complex_mol,
        opts,
    )

    prot_mol = trim_small_chains(prot_mol, prot_cutoff_len)

    return {
        "complex": complex_mol,
        "lig": lig_mol,
        "pro": prot_mol,
        "water": water_mol,
        "other": oth_mol,
    }


def split_openeye_design_unit(du, lig=None, lig_title=None, include_solvent=True):
    """
    Parameters
    ----------
    du : oechem.OEDesignUnit
        Design Unit to be saved
    lig : oechem.OEGraphMol, optional

    lig_title : str, optional
        ID of Ligand to be saved to the Title tag in the SDF, by default None
    include_solvent : bool, optional
        Whether to include solvent in the complex, by default True

    Returns
    -------
    lig : oechem.OEGraphMol
        OE object containing ligand
    protein : oechem.OEGraphMol
        OE object containing only protein
    complex : oechem.OEGraphMol
        OE object containing ligand + protein
    """
    prot = oechem.OEGraphMol()
    complex_ = oechem.OEGraphMol()
    # complex_ = du_to_complex(du, include_solvent=include_solvent)
    du.GetProtein(prot)
    if not lig:
        lig = oechem.OEGraphMol()
        du.GetLigand(lig)

    # Set ligand title, useful for the combined sdf file
    if lig_title:
        lig.SetTitle(f"{lig_title}")

    # Give ligand atoms their own chain "L" and set the resname to "LIG"
    residue = oechem.OEAtomGetResidue(next(iter(lig.GetAtoms())))
    residue.SetChainID("L")
    residue.SetName("LIG")
    residue.SetHetAtom(True)
    for atom in list(lig.GetAtoms()):
        oechem.OEAtomSetResidue(atom, residue)

    # Combine protein and ligand and save
    # TODO: consider saving water as well
    oechem.OEAddMols(complex_, prot)
    oechem.OEAddMols(complex_, lig)

    # Clean up PDB info by re-perceiving, perserving chain ID,
    # residue number, and residue name
    openeye_perceive_residues(prot)
    openeye_perceive_residues(complex_)
    return lig, prot, complex_


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
        oechem.OEDesignUnitComponents_Protein | oechem.OEDesignUnitComponents_Ligand
    )
    if include_solvent:
        comp_tag |= oechem.OEDesignUnitComponents_Solvent
    du.GetComponents(complex_mol, comp_tag)

    complex_mol = openeye_perceive_residues(complex_mol)

    return complex_mol


def trim_small_chains(input_mol, cutoff_len=10):
    """
    Remove short chains from a protein molecule object. The goal is to get rid
    of any peptide ligands that were mistakenly collected by OESplitMolComplex.

    Parameters
    ----------
    input_mol : oechem.OEGraphMol
        OEGraphMol object containing the protein to trim
    cutoff_len : int, default=10
        The cutoff length for peptide chains (chains must have more than this
        many residues to be included)

    Returns
    -------
    oechem.OEGraphMol
        Trimmed molecule
    """
    # Copy the molecule
    mol_copy = input_mol.CreateCopy()

    # Remove chains from mol_copy that are too short (possibly a better way of
    #  doing this with OpenEye functions)
    # Get number of residues per chain
    chain_len_dict = {}
    hv = oechem.OEHierView(mol_copy)
    for chain in hv.GetChains():
        chain_id = chain.GetChainID()
        for frag in chain.GetFragments():
            frag_len = len(list(frag.GetResidues()))
            try:
                chain_len_dict[chain_id] += frag_len
            except KeyError:
                chain_len_dict[chain_id] = frag_len

    # Remove short chains atom by atom
    for a in mol_copy.GetAtoms():
        chain_id = oechem.OEAtomGetResidue(a).GetExtChainID()
        if (chain_id not in chain_len_dict) or (chain_len_dict[chain_id] <= cutoff_len):
            mol_copy.DeleteAtom(a)

    return mol_copy
