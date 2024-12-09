import warnings
from functools import reduce
from pathlib import Path
from typing import Optional, Union

from asapdiscovery.data.backend.openeye import (
    oechem,
    oedocking,
    oegrid,
    oespruce,
    openeye_perceive_residues,
)
from asapdiscovery.modeling.schema import MoleculeComponent, MoleculeFilter


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


def get_oe_prep_opts():
    """
    These are the default options we've been using for OESpruce. They are based on John's function.
    Returns
    -------
    oespruce.OEMakeDesignUnitOptions

    """
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
    return opts


def get_oe_structure_metadata_from_sequence(initial_prot, protein_sequence):
    """
    Add sequence to structure metadata

    Parameters
    ----------
    initial_prot : oechem.OEGraphMol
        The input protein structure to be prepared.
    protein_sequence : str
        The protein sequence to be added to the metadata

    Returns
    -------
    metadata : oespruce.OEStructureMetadata
        The metadata object with the sequence added.
    """
    # Structure metadata object
    metadata = oespruce.OEStructureMetadata()

    # Add SEQRES metadata
    all_prot_chains = find_component_chains(initial_prot, "protein")
    for chain in all_prot_chains:
        seq_metadata = oespruce.OESequenceMetadata()
        seq_metadata.SetChainID(chain)
        seq_metadata.SetSequence(protein_sequence)
        metadata.AddSequenceMetadata(seq_metadata)
    return metadata


def spruce_protein(
    initial_prot: oechem.OEGraphMol,
    protein_sequence: str = None,
    loop_db: Path = None,
) -> oechem.OEDesignUnit or oechem.OEGraphMol:
    """
    Applies the OESpruce protein preparation pipeline to the given protein structure.

    Parameters
    ----------
    initial_prot : oechem.OEMol
        The input protein structure to be prepared.

    protein_sequence : str, optional
        The sequence of the protein for a single change. If provided, this will be added to the Structure Metadata before applying the OESpruce pipeline.
        Default is None.

    loop_db : str, optional
        The filename of the loop database to be used by the OESpruce pipeline. If provided, the pipeline will include the loop building step.
        Default is None.

    Returns
    -------
    (success: bool, spruce_error_msg: str, initial_prot: oechem.OEMol)
        Returns a tuple of:
        a boolean for whether sprucing was successful
        a string of the error message if sprucing failed
        the prepared protein structure.
    """

    # Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)

    opts = get_oe_prep_opts()

    # Set Build Loop and Sidechain Opts
    sc_opts = oespruce.OESidechainBuilderOptions()

    loop_opts = oespruce.OELoopBuilderOptions()
    loop_opts.SetSeqAlignMethod(oechem.OESeqAlignmentMethod_Identity)
    loop_opts.SetSeqAlignGapPenalty(-1)
    loop_opts.SetSeqAlignExtendPenalty(0)

    # Allow for adding residues at the beginning/end if they're missing
    loop_opts.SetBuildTails(True)

    if loop_db is not None:
        print("Adding loop db")
        loop_opts.SetLoopDBFilename(str(loop_db))

    # Construct spruce filter
    spruce_opts = oespruce.OESpruceFilterOptions()
    spruce = oespruce.OESpruceFilter(spruce_opts, opts)

    # Spruce!

    # These objects is for some reason needed in order to run spruce
    grid = oegrid.OESkewGrid()
    if protein_sequence:
        metadata = get_oe_structure_metadata_from_sequence(
            initial_prot, protein_sequence
        )
    else:
        metadata = oespruce.OEStructureMetadata()

    # Building the loops actually does use the sequence metadata
    build_loops_success = oespruce.OEBuildLoops(
        initial_prot, metadata, sc_opts, loop_opts
    )
    build_sidechains_success = oespruce.OEBuildSidechains(initial_prot, sc_opts)
    place_hydrogens_success = oechem.OEPlaceHydrogens(initial_prot)
    spruce_error_code = spruce.StandardizeAndFilter(initial_prot, grid, metadata)
    spruce_error_msg = spruce.GetMessages(spruce_error_code)
    success = (
        build_loops_success and build_sidechains_success and place_hydrogens_success
    )
    # Re-percieve residues so that atom number and connect records dont get screwed up
    initial_prot = openeye_perceive_residues(initial_prot, preserve_all=False)
    return success, spruce_error_msg, initial_prot


def make_design_unit(
    initial_prot: oechem.OEGraphMol,
    site_residue: str = None,
    protein_sequence: str = None,
):
    """

    Parameters
    ----------
    initial_prot: oechem.OEGraphMol
    site_residue: str, optional
        The site residue used to define the binding site of the protein. i.e. "HIS:41: :A".
        This is necessary when there is no ligand in the input structure, otherwise OpenEye will not know where to put the binding site.
    protein_sequence: str, optional

    Returns
    -------

    """
    opts = get_oe_prep_opts()
    if protein_sequence:
        metadata = get_oe_structure_metadata_from_sequence(
            initial_prot, protein_sequence
        )
    else:
        metadata = oespruce.OEStructureMetadata()

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
        success = True
    except IndexError:
        success = False
        du = None
    return success, du


def make_du_from_new_lig(
    protein: oechem.OEGraphMol,
    lig: oechem.OEGraphMol,
    opts: oespruce.OEMakeDesignUnitOptions = None,
):
    """
    Make a design unit from a protein and ligand. Does not resolve clashes,
    and should really only be used to guide docking.

    Parameters
    ----------
    protein : oechem.OEGraphMol
        Protein molecule
    lig : oechem.OEGraphMol
        Ligand molecule
    opts : oechem.OEMakeDesignUnitOptions, optional
        Options for making the design unit, by default the options from `get_oe_prep_opts` will be used.
    """
    if not opts:
        opts = get_oe_prep_opts()
    du = oechem.OEDesignUnit()
    success = oespruce.OEMakeDesignUnit(du, protein, lig, opts)
    return success, du


def superpose_molecule(ref_mol, mobile_mol, ref_chain="A", mobile_chain="A"):
    """
    Superpose `mobile_mol` onto `ref_mol`.

    Parameters
    ----------
    ref_mol : oechem.OEGraphMol
        Reference molecule to align to.
    mobile_mol : oechem.OEGraphMol
        Molecule to align.
    ref_chain : Reference chain to align to
    mobile_chain : Mobile chain to use for alignment (the whole molecule will move as well though)

    Returns
    -------
    oechem.OEGraphMol
        New aligned molecule.
    float
        RMSD between `ref_mol` and `mobile_mol` after alignment.
    """
    chains_in_ref = find_component_chains(ref_mol, "protein", sort_by="size")
    if ref_chain not in chains_in_ref or ref_chain is None:
        warnings.warn(
            f"Chain {ref_chain} not found in reference molecule: chains {chains_in_ref}, using largest chain as reference {chains_in_ref[0]}"
        )
        ref_chain = chains_in_ref[0]

    chains_in_mobile = find_component_chains(mobile_mol, "protein", sort_by="size")
    if mobile_chain not in chains_in_mobile or mobile_chain is None:
        warnings.warn(
            f"Chain {mobile_chain} not found in mobile molecule: chains {chains_in_mobile}, using largest chain {chains_in_mobile[0]}"
        )
        mobile_chain = chains_in_mobile[0]

    if ref_chain != mobile_chain:
        warnings.warn(
            f"Chains {ref_chain} and {mobile_chain} are not the same, this may not be what you want"
        )
    ref_pred = oechem.OEHasChainID(ref_chain)
    mobile_pred = oechem.OEHasChainID(mobile_chain)

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
    openeye_perceive_residues(mut_prot, preserve_all=True)

    return mut_prot


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


def find_component_chains(
    mol: oechem.OEMolBase, component: str, res_name=None, sort_by="alphabetical"
):

    if sort_by not in ["alphabetical", "size"]:
        raise ValueError("sort_by must be either 'alphabetical' or 'size'")

    molcomp = MoleculeComponent(component)

    if res_name:
        # find all chains and their lengths in residues
        chain_lengths = {}
        for res in oechem.OEGetResidues(mol):
            if res.GetName() == res_name:
                chain_id = res.GetChainID()
                if chain_id not in chain_lengths:
                    chain_lengths[chain_id] = 0
                chain_lengths[chain_id] += 1

    elif molcomp.name == MoleculeComponent.PROTEIN.name:
        chain_lengths = {}
        for res in oechem.OEGetResidues(mol):
            if oechem.OEIsStandardProteinResidue(res):
                chain_id = res.GetChainID()
                if chain_id not in chain_lengths:
                    chain_lengths[chain_id] = 0
                chain_lengths[chain_id] += 1

    elif molcomp.name == MoleculeComponent.LIGAND.name:
        chain_lengths = {}
        for res in oechem.OEGetResidues(mol):
            if res.IsHetAtom() and not res.GetName() == "HOH":
                chain_id = res.GetChainID()
                if chain_id not in chain_lengths:
                    chain_lengths[chain_id] = 0
                chain_lengths[chain_id] += 1

    if sort_by == "size":
        chainids = sorted(
            chain_lengths.keys(), key=lambda x: chain_lengths[x], reverse=True
        )
    elif sort_by == "alphabetical":
        chainids = sorted(chain_lengths.keys())

    return chainids


def split_openeye_mol(
    complex_mol,
    molecule_filter: Optional[Union[str, list[str], MoleculeFilter]] = None,
    prot_cutoff_len=10,
    keep_one_lig=True,
) -> dict:
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
    prot_cutoff_len : int, default=10
        Minimum number of residues in a protein chain required in order to keep
    keep_one_lig : bool, default=True
        Only keep one copy of the ligand in lig_mol

    Returns
    -------
    dict
        A dict containing the protein ("prot"), ligand ("lig"), water atoms ("wat"),
        and all other atoms ("oth")
    """
    # These are the objects that will be returned at the end
    lig_mol = oechem.OEGraphMol()
    prot_mol = oechem.OEGraphMol()
    water_mol = oechem.OEGraphMol()
    oth_mol = oechem.OEGraphMol()

    if molecule_filter is None:
        molecule_filter = MoleculeFilter()
    elif type(molecule_filter) is str:
        molecule_filter = MoleculeFilter(components_to_keep=[molecule_filter])
    elif type(molecule_filter) is list:
        molecule_filter = MoleculeFilter(components_to_keep=molecule_filter)
    else:
        molecule_filter = molecule_filter

    # Make splitting split out covalent ligands possible
    # TODO: look into different covalent-related options here
    opts = oechem.OESplitMolComplexOptions()
    opts.SetSplitCovalent(True)
    opts.SetSplitCovalentCofactors(True)

    # Protein splitting options
    # Default of all atoms identified as protein + complex
    prot_only = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_ProtComplex
    )
    # add in peptides as well, sometimes proteins are misidentified as peptides if bound or short
    peptide = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_Peptide
    )

    # combine protein and peptide filters
    prot_only = oechem.OEOrRoleSet(prot_only, peptide)

    # If protein_chains are specified, only take protein atoms from those chains
    if len(molecule_filter.protein_chains) > 0:
        chain_filters = [
            oechem.OERoleMolComplexFilterFactory(
                oechem.OEMolComplexChainRoleFactory(chain)
            )
            for chain in molecule_filter.protein_chains
        ]

        chain_filter = reduce(oechem.OEOrRoleSet, chain_filters)
        prot_filter = oechem.OEAndRoleSet(prot_only, chain_filter)
    else:
        prot_filter = prot_only
    opts.SetProteinFilter(prot_filter)

    # Ligand splitting options
    # Default of all atoms identified as ligand
    lig_only = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_Ligand
    )
    # If ligand_chain is specified, only take protein atoms from that chains
    if molecule_filter.ligand_chain:
        lig_chain = oechem.OERoleMolComplexFilterFactory(
            oechem.OEMolComplexChainRoleFactory(molecule_filter.ligand_chain)
        )
        lig_filter = oechem.OEAndRoleSet(lig_only, lig_chain)
    else:
        lig_filter = lig_only
    # combine with NOT peptide filter
    lig_filter = oechem.OEAndRoleSet(lig_filter, oechem.OENotRoleSet(peptide))
    opts.SetLigandFilter(lig_filter)

    # If water_chains are specified, set up filter for them
    if len(molecule_filter.water_chains) > 0:
        water_only = oechem.OEMolComplexFilterFactory(
            oechem.OEMolComplexFilterCategory_Water
        )
        chain_filters = [
            oechem.OERoleMolComplexFilterFactory(
                oechem.OEMolComplexChainRoleFactory(chain)
            )
            for chain in molecule_filter.water_chains
        ]

        chain_filter = reduce(oechem.OEOrRoleSet, chain_filters)
        wat_filter = oechem.OEAndRoleSet(water_only, chain_filter)
        # combine with NOT peptide filter
        wat_filter = oechem.OEAndRoleSet(wat_filter, oechem.OENotRoleSet(peptide))
        opts.SetWaterFilter(wat_filter)

    # Use python 'reduce' to combine all the filters into one, otherwise OpenEye will throw an error
    oechem.OESplitMolComplex(
        lig_mol,
        prot_mol,
        water_mol,
        oth_mol,
        complex_mol,
        opts,
    )
    # TODO: make this nicer?
    # Get rid of any straggling extra copies of the ligand
    prot_mol = trim_small_chains(prot_mol, prot_cutoff_len)

    if molecule_filter.ligand_chain:
        keep_lig_chain = molecule_filter.ligand_chain
    else:
        keep_lig_chain = None
    # Get rid of extra copies of the ligand
    if keep_one_lig:
        all_lig_chains = [res.GetChainID() for res in oechem.OEGetResidues(lig_mol)]
        # Handle the case where the input has no ligand, otherwise throws an IndexError
        if len(all_lig_chains) > 0:
            # Keep first alphabetically, for reproducibility
            if not keep_lig_chain:
                keep_lig_chain = sorted(all_lig_chains)[0]
            for a in lig_mol.GetAtoms():
                # Delete all atoms that don't match
                if oechem.OEAtomGetResidue(a).GetChainID() != keep_lig_chain:
                    lig_mol.DeleteAtom(a)

    return {
        "prot": prot_mol,
        "lig": lig_mol,
        "wat": water_mol,
        "oth": oth_mol,
        "keep_lig_chain": keep_lig_chain,
    }


def split_openeye_design_unit(du, lig=None, lig_title=None):
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

    du.GetProtein(prot)

    # if no ligand, return protein and complex with no ligand
    if not du.HasLigand():
        return None, prot, prot

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
