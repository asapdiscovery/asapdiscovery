"""
Create oedu binary DesignUnit files for input protein structures, with all methods copied into this script in order
to make it a single script for the COVID Moonshot paper. This script is copied in the covid-moonshot-manuscript but
just in case it is saved here.
It is not intended to be an entry point in this repo or maintained with updated methods.

Example Usage:
    python prep_proteins.py \
    -d mpro_fragalysis_2022_10_12/aligned/ \
    -x mpro_fragalysis_2022_10_12/metadata.csv \
    -o full_frag_prepped_mpro_20230125/ \
    -l rcsb_spruce.loop_db \
    -n 32 \
    -s covid-moonshot-ml/metadata/mpro_sars2_seqres.yaml \
    --include_non_Pseries
"""
import argparse
import datetime
import logging
import multiprocessing as mp
import os

import yaml
from openeye import oechem, oedocking, oespruce
from pydantic import BaseModel, Field


########################################################################################################################
# CODE THAT WAS PREVIOUSLY IN IMPORTED METHODS
# from asapdiscovery.data.schema import CrystalCompoundData
# from asapdiscovery.data import pdb
# from asapdiscovery.data.utils import seqres_to_res_list
# from asapdiscovery.data.openeye import (
#     save_openeye_pdb,
#     load_openeye_pdb,
# )
# from asapdiscovery.data.fragalysis import parse_fragalysis
# from asapdiscovery.docking.modeling import (
#     align_receptor,
#     prep_receptor,
#     du_to_complex,
#     mutate_residues,
#     remove_extra_ligands,
# )
########################################################################################################################
class CrystalCompoundData(BaseModel):
    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    compound_id: str = Field(
        None, description="The unique compound identifier of the ligand."
    )

    dataset: str = Field(
        None, description="Dataset name from Fragalysis (name of structure)."
    )

    str_fn: str = Field(None, description="Filename of the PDB structure.")

    sdf_fn: str = Field(None, description="Filename of the SDF file")
    active_site_chain: str = Field(
        None, description="Chain identifying the active site of interest."
    )
    output_name: str = Field(None, description="Name of output structure.")
    active_site: str = Field(None, description="OpenEye formatted active site residue.")
    oligomeric_state: str = Field(
        None, description="Oligomeric state of the asymmetric unit."
    )
    chains: list = Field(None, description="List of chainids in the asymmetric unit.")
    protein_chains: list = Field(
        None, description="List of chains corresponding to protein residues."
    )


def save_openeye_design_unit(du, lig=None, lig_title=None):
    """
    Parameters
    ----------
    du : oechem.OEDesignUnit
        Design Unit to be saved

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
    complex = oechem.OEGraphMol()
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
    oechem.OEAddMols(complex, prot)
    oechem.OEAddMols(complex, lig)

    # Clean up PDB info by re-perceiving, perserving chain ID, residue number, and residue name
    openeye_perceive_residues(prot)
    return lig, prot, complex


def load_openeye_pdb(pdb_fn, alt_loc=False):
    ifs = oechem.oemolistream()
    ifs_flavor = oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA
    # Add option for keeping track of alternat locations in PDB file
    if alt_loc:
        ifs_flavor |= oechem.OEIFlavor_PDB_ALTLOC
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        ifs_flavor,
    )
    ifs.open(pdb_fn)
    in_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, in_mol)
    ifs.close()

    return in_mol


def load_openeye_sdf(sdf_fn):
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    ifs.open(sdf_fn)
    coords_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, coords_mol)
    ifs.close()

    return coords_mol


def save_openeye_pdb(mol, pdb_fn):
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_PDB, oechem.OEOFlavor_PDB_Default)
    ofs.open(pdb_fn)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def save_openeye_sdf(mol, sdf_fn):
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_SDF, oechem.OEOFlavor_SDF_Default)
    ofs.open(sdf_fn)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def save_openeye_sdfs(mols, sdf_fn):
    """
    Parameters
    ----------
    mols: list of OEGraphMol
    sdf_fn: str
        SDF file path
    """
    ofs = oechem.oemolostream()
    ofs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEOFlavor_SDF_Default,
    )
    if ofs.open(sdf_fn):
        for mol in mols:
            oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")


def openeye_perceive_residues(prot: oechem.OEGraphMol) -> oechem.OEGraphMol:
    """
    Function for doing basic openeye percieve residues function,
    necessary when changes are made to the protein to ensure correct atom ordering and CONECT record creation

    Parameters
    ----------
    prot: oechem.OEGraphMol

    Returns
    -------
    prot: oechem.OEGraphMol

    """
    # Clean up PDB info by re-perceiving, perserving chain ID, residue number, and residue name
    preserve = (
        oechem.OEPreserveResInfo_ChainID
        | oechem.OEPreserveResInfo_ResidueNumber
        | oechem.OEPreserveResInfo_ResidueName
    )
    oechem.OEPerceiveResidues(prot, preserve)

    return prot


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


def split_openeye_mol(complex_mol, lig_chain="A", prot_cutoff_len=10):
    """
    Split an OpenEye-loaded molecule into protein, ligand, etc.

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
    Basically a copy of the above function to generate an aligned receptor without also needing to do the rest of the
    protein prep.

    Parameters
    ----------
    initial_complex : Union[oechem.OEGraphMol, str]
        Initial complex loaded straight from a PDB file. Can contain ligands,
        waters, cofactors, etc., which will be removed. Can also pass a PDB
        filename instead.
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
            # If the residue number is out of range (because its a water or something weird)
            # then we can skip right on by it
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

    # Options set from John's function ########################################
    # (https://github.com/FoldingAtHome/covid-moonshot/blob/454098f4255467f4655102e0330ebf9da0d09ccb/synthetic-enumeration/sprint-14-quinolones/00-prep-receptor.py)
    opts.GetPrepOptions().SetStrictProtonationMode(True)
    # set minimal number of ligand atoms to 5, e.g. a 5-membered ring fragment\
    opts.GetSplitOptions().SetMinLigAtoms(5)

    # also consider alternate locations outside binding pocket, important for later filtering
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

    ############################################################################

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

    # Detect ligand chain to keep if none is given
    if lig_chain is None:
        lig_chain = sorted(
            {
                oechem.OEAtomGetResidue(a).GetExtChainID()
                for a in mol.GetAtoms(all_lig_filter)
            }
        )[0]

    # Copy molecule and delete all lig atoms that don't have the desired chain
    mol_copy = mol.CreateCopy()
    for a in mol_copy.GetAtoms(all_lig_filter):
        if oechem.OEAtomGetResidue(a).GetExtChainID() != lig_chain:
            mol_copy.DeleteAtom(a)
    return mol_copy


def seqres_to_res_list(seqres_str):
    """
    https://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#SEQRES
    Parameters
    ----------
    seqres_str

    Returns
    -------

    """
    # Grab the sequence from the sequence str
    # use chain ID column
    seqres_chain_column = 11
    seq_lines = [
        line[19:]
        for line in seqres_str.split("\n")
        if len(line) > 0
        if line[seqres_chain_column] == "A"
    ]
    seq_str = " ".join(seq_lines)
    res_list = seq_str.split(" ")
    return res_list


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

    return complex_mol


def parse_fragalysis(
    x_fn,
    x_dir,
    name_filter=None,
    name_filter_column="crystal_name",
    drop_duplicate_datasets=False,
):
    """
    Load all crystal structures into schema.CrystalCompoundData objects.
    Parameters
    ----------
    x_fn : str
        metadata.CSV file giving information on each crystal structure
    x_dir : str
        Path to directory containing directories with crystal structure PDB
        files
    name_filter : str or list
        String or list of strings that are required to be in the name_filter_column
    name_filter_column : str
        Name of column in the metadata.csv that will be used to filter the dataframe
    drop_duplicate_datasets : bool
        If true, will drop the _1A, _0B, etc duplicate datasets for a given crystal structure.
    Returns
    -------
    List[schema.CrystalCompoundData]
        List of parsed crystal structures
    """
    from pathlib import Path

    import pandas
    from tqdm import tqdm

    x_dir = Path(x_dir)

    df = pandas.read_csv(x_fn)

    # Only keep rows of dataframe where the name_filter_column includes the name_filter string
    if name_filter:
        if type(name_filter) == str:
            idx = df[name_filter_column].apply(lambda x: name_filter in x)
            df = df[idx]
        elif type(name_filter) == list:
            for filter in name_filter:
                idx = df[name_filter_column].apply(lambda x: filter in x)
                df = df[idx]
    # Drop duplicates, keeping only the first one.
    if drop_duplicate_datasets:
        df = df.drop_duplicates("RealCrystalName")

    # Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [
        dict(zip(("smiles", "dataset", "compound_id"), r[1].values))
        for r in df.loc[:, ["smiles", "crystal_name", "alternate_name"]].iterrows()
    ]

    # Add structure filename information and filter if not found
    filtered_xtal_dicts = []
    for d in tqdm(xtal_dicts):
        glob_str = f"{d['dataset']}*/*.pdb"
        fns = list(x_dir.glob(glob_str))
        for fn in fns:
            d["str_fn"] = str(fn)

            # This should basically always be true since we're getting the filenames from glob but just in case.
            if os.path.isfile(fn):
                filtered_xtal_dicts.append(d)
    assert (
        len(filtered_xtal_dicts) > 0
    ), "No structure filenames were found by parse_fragalysis"

    # Build CrystalCompoundData objects for each row
    print(f"Loading {len(filtered_xtal_dicts)} structures")
    xtal_compounds = [CrystalCompoundData(**d) for d in filtered_xtal_dicts]

    return xtal_compounds


########################################################################################################################
########################################################################################################################


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
    # Make output directory
    out_dir = os.path.join(out_base, f"{xtal.output_name}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Prepare logger
    handler = logging.FileHandler(os.path.join(out_dir, "log.txt"), mode="w")
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
        save_openeye_pdb(initial_prot, "align_test.pdb")
    # Take the first returned DU and save it
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


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to fragalysis/aligned/ directory or directory to put PDB structures.",
    )

    parser.add_argument(
        "-x",
        "--xtal_csv",
        default=None,
        help="CSV file giving information of which structures to prep.",
    )

    parser.add_argument(
        "-r",
        "--ref_prot",
        default=None,
        type=str,
        help="Path to reference pdb to align to. If None, no alignment will be performed",
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    # Model-building arguments
    parser.add_argument(
        "-l",
        "--loop_db",
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "--protein_only",
        action="store_true",
        default=False,
        help="If true, generate design units with only the protein in them",
    )
    parser.add_argument(
        "--include_non_Pseries",
        default=False,
        action="store_true",
        help="If true, the p_only flag of parse_xtal will be set to False. Default is False, which sets p_only to True",
    )
    parser.add_argument(
        "--log_file",
        default="prep_proteins_log.txt",
        help="Path to high level log file.",
    )

    # Performance arguments
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    handler = logging.FileHandler(args.log_file, mode="w")
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(handler)

    p_only = False if args.include_non_Pseries else True
    if p_only:
        xtal_compounds = parse_fragalysis(
            args.xtal_csv,
            args.structure_dir,
            name_filter="Mpro-P",
            drop_duplicate_datasets=True,
        )
    else:
        xtal_compounds = parse_fragalysis(
            args.xtal_csv,
            args.structure_dir,
        )

    for xtal in xtal_compounds:
        # Get chain
        # The parentheses in this string are the capture group

        xtal.output_name = f"{xtal.dataset}_{xtal.compound_id}"

        frag_chain = xtal.dataset[-2:]

        # We also want the chain in the form of a single letter ('A', 'B'), etc
        xtal.active_site_chain = frag_chain[-1]

        # If we aren't keeping the ligands, then we want to give it a site residue to use
        if args.protein_only:
            xtal.active_site = f"His:41: :{xtal.active_site_chain}"

    if args.seqres_yaml:
        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]
    else:
        seqres = None

    mp_args = [
        (
            x,
            args.ref_prot,
            seqres,
            args.output_dir,
            args.loop_db,
            args.protein_only,
        )
        for x in xtal_compounds
    ]
    main_logger.info(mp_args[0])
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    main_logger.info(
        f"CPUS: {mp.cpu_count()}, Structure: {mp_args}, N Cores: {args.num_cores}"
    )
    main_logger.info(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
