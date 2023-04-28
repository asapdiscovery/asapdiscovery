from pathlib import Path

from openeye import oechem, oedepict, oedocking, oegrid, oeomega, oespruce  # noqa: F401

# exec on module import
if not oechem.OEChemIsLicensed("python"):
    raise RuntimeError("OpenEye license required to use asapdiscovery openeye module")


def combine_protein_ligand(prot, lig, lig_name="LIG", resid=None, start_atom_id=None):
    """
    Combine a protein OEMol and ligand OEMol into one, handling residue/atom
    numbering, and HetAtom status.

    Parameters
    ----------
    prot : oechem.OEMol
        OEMol with the protein atoms. This should have perceived resiudes
    lig : oechem.OEMol
        OEMol with the ligand atoms
    lig_name : str, default="LIG"
        Residue name to give to the ligand atoms
    resid : int, optional
        Which residue number to assign to the ligand. If not given, the largest existing
        residue number in `prot` will be found, and the ligand will be assigned the next
        number
    start_atom_id : int, optional
        Which atom number to assign to the first atom in the ligand. If not given, the
        next available atom number will be calculated and assigned

    Returns
    -------
    oechem.OEMol
        Combined molecule, with the appropriate biopolymer field set for the lig atoms
    """
    # Calculate residue number if necessary
    if resid is None:
        # Find max resid for numbering the ligand residue
        # Add 1 so we start numbering at the next residue id
        resid = max([r.GetResidueNumber() for r in oechem.OEGetResidues(prot)]) + 1

    # Calculate atom number if necessary
    if start_atom_id is None:
        # Same with atom numbering
        start_atom_id = (
            max([oechem.OEAtomGetResidue(a).GetSerialNumber() for a in prot.GetAtoms()])
            + 1
        )

    # Make copies so we don't modify the original molecules
    prot = prot.CreateCopy()
    lig = lig.CreateCopy()

    # Keep track of how many times each element has been seen in the ligand
    # Each atom in a residue needs a unique name, so just append this number to the
    #  element
    num_elem_atoms = {}
    # Adjust molecule residue properties
    for a in lig.GetAtoms():
        # Set atom name
        cur_name = oechem.OEGetAtomicSymbol(a.GetAtomicNum())
        try:
            new_name = f"{cur_name}{num_elem_atoms[cur_name]}"
            num_elem_atoms[cur_name] += 1
        except KeyError:
            new_name = cur_name
            num_elem_atoms[cur_name] = 1
        a.SetName(new_name)

        # Set residue level properties
        res = oechem.OEAtomGetResidue(a)
        res.SetName(lig_name.upper())
        res.SetResidueNumber(resid)
        res.SetSerialNumber(start_atom_id)
        start_atom_id += 1
        res.SetHetAtom(True)
        oechem.OEAtomSetResidue(a, res)

    # Combine the mols
    oechem.OEAddMols(prot, lig)

    return prot


def load_openeye_pdb(pdb_fn, alt_loc=False):
    """
    Load an OpenEye OEGraphMol from a PDB file.

    Parameters
    ----------
    pdb_fn : str
        The path to the input PDB file.
    alt_loc : bool, optional
        Whether to keep track of alternate locations, by default False.

    Returns
    -------
    oechem.OEGraphMol
        The OEGraphMol loaded from the input file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    oechem.OEError
        If the CIF file cannot be opened.
    """
    if not Path(pdb_fn).exists():
        raise FileNotFoundError(f"{pdb_fn} does not exist!")
    ifs = oechem.oemolistream()
    ifs_flavor = oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA
    # Add option for keeping track of alternat locations in PDB file
    if alt_loc:
        ifs_flavor |= oechem.OEIFlavor_PDB_ALTLOC
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        ifs_flavor,
    )
    if ifs.open(pdb_fn):
        in_mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, in_mol)
        ifs.close()
        return in_mol

    else:
        oechem.OEThrow.Fatal(f"Unable to open {pdb_fn}")


def load_openeye_cif(cif_fn, alt_loc=False):
    """
    Load an OpenEye OEGraphMol object from a CIF file.

    Parameters
    ----------
    cif_fn : str
        The path of the CIF file to read.
    alt_loc : bool, optional
        If True, include alternative locations for atoms in the resulting OEGraphMol
        object. Default is False.

    Returns
    -------
    oechem.OEGraphMol
        The OEGraphMol object read from the CIF file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    oechem.OEError
        If the CIF file cannot be opened.

    Notes
    -----
    This function will raise an exception if the specified file does not exist or
    if the CIF file cannot be opened. If `alt_loc` is True, the resulting OEGraphMol
    object will include alternative locations for atoms.
    """
    if not Path(cif_fn).exists():
        raise FileNotFoundError(f"{cif_fn} does not exist!")

    ifs = oechem.oemolistream()
    ifs_flavor = oechem.OEIFlavor_MMCIF_DEFAULT
    # Add option for keeping track of alternat locations in PDB file
    # TODO: check if this is a thing in mmcif
    # TODO: Currently this actually fails on biological assemblies so I'm not using it but it *should* work
    if not alt_loc:
        ifs_flavor |= oechem.OEIFlavor_MMCIF_NoAltLoc
    ifs.SetFlavor(
        oechem.OEFormat_MMCIF,
        ifs_flavor,
    )
    if ifs.open(cif_fn):
        in_mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, in_mol)
        ifs.close()
        return in_mol

    else:
        oechem.OEThrow.Fatal(f"Unable to open {cif_fn}")


def load_openeye_sdf(sdf_fn) -> oechem.OEGraphMol:
    """
    Load an OpenEye SDF file containing a single molecule and return it as an
    OpenEye OEGraphMol object.

    Parameters
    ----------
    sdf_fn : str
        Path to the SDF file to load.

    Returns
    -------
    oechem.OEGraphMol
        An OpenEye OEGraphMol object containing the molecule data from the SDF file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    oechem.OEError
        If the CIF file cannot be opened.

    Notes
    -----
    This function assumes that the SDF file contains a single molecule. If the
    file contains more than one molecule, only the first molecule will be loaded.
    """

    if not Path(sdf_fn).exists():
        raise FileNotFoundError(f"{sdf_fn} does not exist!")

    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    if ifs.open(sdf_fn):
        coords_mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, coords_mol)
        ifs.close()
        return coords_mol
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")


def load_openeye_sdfs(sdf_fn):
    """
    Load a list of OpenEye OEGraphMol objects from an SDF file.

    Parameters
    ----------
    sdf_fn : str
        The path of the SDF file to read.

    Returns
    -------
    list of oechem.OEGraphMol
        A list of the OEGraphMol objects read from the SDF file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    oechem.OEError
        If the CIF file cannot be opened.

    Notes
    -----
    This function will raise an exception if the specified file does not exist or
    if the SDF file cannot be opened.
    """
    if not Path(sdf_fn).exists():
        raise FileNotFoundError(f"{sdf_fn} does not exist!")
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    cmpd_list = []
    if ifs.open(sdf_fn):
        for mol in ifs.GetOEGraphMols():
            cmpd_list.append(mol.CreateCopy())
        ifs.close()
        return cmpd_list
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")


def save_openeye_pdb(mol, pdb_fn):
    """
    Write an OpenEye OEGraphMol object to a PDB file.

    Parameters
    ----------
    mol : oechem.OEGraphMol
        The OEGraphMol object to write to the PDB file.
    pdb_fn : str
        The path of the PDB file to create or overwrite.

    Returns
    -------
    None

    Notes
    -----
    This function will overwrite any existing file with the same name as `pdb_fn`.
    """
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_PDB, oechem.OEOFlavor_PDB_Default)
    ofs.open(pdb_fn)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def save_openeye_sdf(mol, sdf_fn):
    """
    Write an OpenEye OEGraphMol object to an SDF file.

    Parameters
    ----------
    mol : oechem.OEGraphMol
        The OEGraphMol object to write to the SDF file.
    sdf_fn : str
        The path of the SDF file to create or overwrite.

    Returns
    -------
    None

    Notes
    -----
    This function will overwrite any existing file with the same name as `sdf_fn`.
    """
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_SDF, oechem.OEOFlavor_SDF_Default)
    ofs.open(sdf_fn)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def save_openeye_sdfs(mols, sdf_fn):
    """
    Write a list of OpenEye OEGraphMol objects to a single SDF file.

    Parameters
    ----------
    mols : list of oechem.OEGraphMol
        The list of OEGraphMol objects to write to the SDF file.
    sdf_fn : str
        The path of the SDF file to create or overwrite.

    Returns
    -------
    None

    Raises
    ------
    oechem.OEError
        If the SDF file cannot be opened.

    Notes
    -----
    This function will overwrite any existing file with the same name as `sdf_fn`.
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
    Re-perceive the residues of a protein molecule using OpenEye's OEPerceiveResidues function,
    which is necessary when changes are made to the protein to ensure correct atom ordering and CONECT record creation.

    Parameters
    ----------
    prot : oechem.OEGraphMol
        The input protein molecule to be processed.

    Returns
    -------
    oechem.OEGraphMol
        The processed protein molecule with re-perceived residue information.
    """
    # Clean up PDB info by re-perceiving, perserving chain ID, residue number, and residue name
    preserve = (
        oechem.OEPreserveResInfo_ChainID
        | oechem.OEPreserveResInfo_ResidueNumber
        | oechem.OEPreserveResInfo_ResidueName
    )
    oechem.OEPerceiveResidues(prot, preserve)

    return prot


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
    from .utils import trim_small_chains

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


def get_ligand_rmsd_from_pdb_and_sdf(ref_path, mobile_path, fetch_docking_results=True):
    """
    TODO: This should be deprecated in favor of the functions in docking.analysis
    Calculates the RMSD between a reference ligand from a PDB file and a mobile ligand from an SDF file.
    If `fetch_docking_results` is True, additional docking results from the mobile ligand are returned as a dictionary.

    Parameters
    ----------
    ref_path: str
        Path to the reference PDB file containing the ligand.
    mobile_path: str
        Path to the SDF file containing the mobile ligand.
    fetch_docking_results: bool, optional
        If True, additional docking results from the mobile ligand are returned. Default is True.

    Returns
    -------
    return_dict: dict
        A dictionary with the following keys:
            - 'rmsd': the RMSD between the reference and mobile ligands.
            - 'posit': (if `fetch_docking_results` is True) the docking score from the mobile ligand's 'POSIT::Probability' SD tag.
            - 'chemgauss': (if `fetch_docking_results` is True) the docking score from the mobile ligand's 'Chemgauss4' SD tag.
    """
    ref_pdb = load_openeye_pdb(ref_path)
    ref = split_openeye_mol(ref_pdb)["lig"]
    mobile = load_openeye_sdf(mobile_path)

    for a in mobile.GetAtoms():
        if a.GetAtomicNum() == 1:
            mobile.DeleteAtom(a)

    rmsd = oechem.OERMSD(ref, mobile)

    return_dict = {"rmsd": rmsd}

    if fetch_docking_results:
        return_dict["posit"] = oechem.OEGetSDData(mobile, "POSIT::Probability")
        return_dict["chemgauss"] = oechem.OEGetSDData(mobile, "Chemgauss4")

    return return_dict


def split_openeye_design_unit(du, lig=None, lig_title=None):
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

    # Clean up PDB info by re-perceiving, perserving chain ID,
    # residue number, and residue name
    openeye_perceive_residues(prot)
    return lig, prot, complex


def save_receptor_grid(du_fn, out_fn):
    """
    Load in a design unit from a file and write out the receptor grid as a .ccp4 grid file.
    Parameters
    ----------
    du_fn: str
        File name/path with the design units.
    out_fn: str
        Works with a .ccp4 extension

    Returns
    -------

    """
    du = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(du_fn, du)
    # oedocking.OEMakeReceptor(du)
    oegrid.OEWriteGrid(
        out_fn,
        oegrid.OEScalarGrid(du.GetReceptor().GetNegativeImageGrid()),
    )


def openeye_copy_pdb_data(
    source: oechem.OEGraphMol, destination: oechem.OEGraphMol, tag: str
):
    """
    Copy over the PDB data from one object to another. Tag examples include "SEQRES"

    Parameters
    ----------
    source: oechem.OEGraphMol
        Source molecule to copy the data from
    destination: oechem.OEGraphMol
        Destination molecule to copy the data
    tag: str
        Tag identifier for the data/metadata.

    Returns
    -------

    """
    # first, delete data with that tag
    oechem.OEDeletePDBData(destination, tag)

    # now, add over all the data with the tag
    for data_pair in oechem.OEGetPDBDataPairs(source):
        if data_pair.GetTag() == tag:
            oechem.OEAddPDBData(destination, data_pair)
