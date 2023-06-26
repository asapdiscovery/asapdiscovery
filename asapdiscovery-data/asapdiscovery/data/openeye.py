from pathlib import Path
from typing import Dict, List, Optional, Union  # noqa: F401

from openeye import oechem, oedepict, oedocking, oegrid, oeomega, oespruce  # noqa: F401

# exec on module import

if not oechem.OEChemIsLicensed("python"):
    raise RuntimeError("OpenEye license required to use asapdiscovery openeye module")


def combine_protein_ligand(
    prot: oechem.OEMol,
    lig: oechem.OEMol,
    lig_name: str = "LIG",
    resid: Optional[int] = None,
    start_atom_id: Optional[int] = None,
) -> oechem.OEMol:
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


def load_openeye_pdb(
    pdb_fn: Union[str, Path], alt_loc: bool = False
) -> oechem.OEGraphMol:
    """
    Load an OpenEye OEGraphMol from a PDB file.

    Parameters
    ----------
    pdb_fn : Union[str, Path]
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
    if ifs.open(str(pdb_fn)):
        in_mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, in_mol)
        ifs.close()
        return in_mol

    else:
        oechem.OEThrow.Fatal(f"Unable to open {pdb_fn}")


def load_openeye_cif1(cif1_fn: Union[str, Path]) -> oechem.OEGraphMol:
    """
    Loads a biological assembly file into an OEGraphMol object.
    Current version requires going through an OpenMM intermediate.

    Parameters
    ----------
    cif1_fn : Union[str, Path]
        The path to the input CIF1 file.

    Returns
    -------
    oechem.OEGraphMol
        oechem.OEGraphMol: the biological assembly as an OEGraphMol object.
    """
    from tempfile import NamedTemporaryFile

    from openmm.app import PDBFile, PDBxFile

    if not Path(cif1_fn).exists():
        raise FileNotFoundError(f"{cif1_fn} does not exist!")

    cif = PDBxFile(str(cif1_fn))

    # the keep ids flag is critical to make sure the residue numbers are correct
    with NamedTemporaryFile("w", suffix=".pdb") as f:
        PDBFile.writeFile(cif.topology, cif.positions, f, keepIds=True)
        prot = load_openeye_pdb(f.name)
    return prot


def load_openeye_cif(
    cif_fn: Union[str, Path], alt_loc: bool = False
) -> oechem.OEGraphMol:
    """
    Load an OpenEye OEGraphMol object from a CIF file.

    Parameters
    ----------
    cif_fn : Union[str, Path]
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
    if ifs.open(str(cif_fn)):
        in_mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, in_mol)
        ifs.close()
        return in_mol

    else:
        oechem.OEThrow.Fatal(f"Unable to open {cif_fn}")


def load_openeye_sdf(sdf_fn: Union[str, Path]) -> oechem.OEGraphMol:
    """
    Load an OpenEye SDF file containing a single molecule and return it as an
    OpenEye OEGraphMol object.

    Parameters
    ----------
    sdf_fn : Union[str, Path]
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
    if ifs.open(str(sdf_fn)):
        coords_mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, coords_mol)
        ifs.close()
        return coords_mol
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")


def load_openeye_sdfs(sdf_fn: Union[str, Path]) -> list[oechem.OEGraphMol]:
    """
    Load a list of OpenEye OEGraphMol objects from an SDF file.

    Parameters
    ----------
    sdf_fn : Union[str, Path]
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
    if ifs.open(str(sdf_fn)):
        for mol in ifs.GetOEGraphMols():
            cmpd_list.append(mol.CreateCopy())
        ifs.close()
        return cmpd_list
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")


def load_openeye_design_unit(du_fn: Union[str, Path]) -> oechem.OEDesignUnit:
    """
    Load an OpenEye design unit from a file

    Parameters
    ----------
    du_fn : Union[str, Path]
        The path of the DesignUnit file to read.

    Returns
    -------
    oechem.OEDesignUnit
        OpenEye DesignUnit

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    oechem.OEError
        If the CIF file cannot be opened.


    """
    if not Path(du_fn).exists():
        raise FileNotFoundError(f"{du_fn} does not exist!")
    du = oechem.OEDesignUnit()
    retcode = oechem.OEReadDesignUnit(str(du_fn), du)
    if not retcode:
        oechem.OEThrow.Fatal(f"Unable to open {du_fn}")
    return du


def save_openeye_pdb(mol, pdb_fn: Union[str, Path]) -> Path:
    """
    Write an OpenEye OEGraphMol object to a PDB file.

    Parameters
    ----------
    mol : oechem.OEGraphMol
        The OEGraphMol object to write to the PDB file.
    pdb_fn : Union[str, Path]
        The path of the PDB file to create or overwrite.

    Returns
    -------
    Path
        The path of the PDB file that was written.

    Notes
    -----
    This function will overwrite any existing file with the same name as `pdb_fn`.
    """
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_PDB, oechem.OEOFlavor_PDB_Default)
    if ofs.open(str(pdb_fn)):
        oechem.OEWriteMolecule(ofs, mol)
    else:
        oechem.OEThrow.Fatal(f"Unable to open {pdb_fn}")
    ofs.close()

    return Path(pdb_fn)


def save_openeye_sdf(mol, sdf_fn: Union[str, Path]) -> Path:
    """
    Write an OpenEye OEGraphMol object to an SDF file.

    Parameters
    ----------
    mol : oechem.OEGraphMol
        The OEGraphMol object to write to the SDF file.
    sdf_fn :  Union[str, Path]
        The path of the SDF file to create or overwrite.

    Returns
    -------
    Path
        The path of the SDF file that was written.

    Notes
    -----
    This function will overwrite any existing file with the same name as `sdf_fn`.
    """
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_SDF, oechem.OEOFlavor_SDF_Default)
    if ofs.open(str(sdf_fn)):
        oechem.OEWriteMolecule(ofs, mol)
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")
    ofs.close()

    return Path(sdf_fn)


def save_openeye_sdfs(mols, sdf_fn: Union[str, Path]) -> Path:
    """
    Write a list of OpenEye OEGraphMol objects to a single SDF file.

    Parameters
    ----------
    mols : list of oechem.OEGraphMol
        The list of OEGraphMol objects to write to the SDF file.
    sdf_fn :  Union[str, Path]
        The path of the SDF file to create or overwrite.

    Returns
    -------
    Path
        The path of the SDF file that was written.

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
    if ofs.open(str(sdf_fn)):
        for mol in mols:
            oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")

    return Path(sdf_fn)


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


def save_receptor_grid(du_fn: Union[str, Path], out_fn: Union[str, Path]) -> Path:
    """
    Load in a design unit from a file and write out the receptor grid as a .ccp4 grid file.
    Parameters
    ----------
    du_fn: Union[str, Path]
        File name/path with the design units.
    out_fn: Union[str, Path]
        Works with a .ccp4 extension

    Returns
    -------

    """
    du = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(du_fn), du)
    # oedocking.OEMakeReceptor(du)
    oegrid.OEWriteGrid(
        str(out_fn),
        oegrid.OEScalarGrid(du.GetReceptor().GetNegativeImageGrid()),
    )

    return Path(out_fn)


def openeye_copy_pdb_data(
    source: oechem.OEGraphMol, destination: oechem.OEGraphMol, tag: str
) -> None:
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


def oemol_to_sdf_string(mol: oechem.OEMol) -> str:
    """
    Dumps an OpenEye OEMol to an SDF string

    Parameters
    ----------
    mol: oechem.OEMol
       OpenEye OEMol

    Returns
    -------
    str
       SDF string representation of the input OEMol
    """
    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_SDF)
    oms.openstring()
    oechem.OEWriteMolecule(oms, mol)
    molstring = oms.GetString().decode("UTF-8")
    return molstring


def sdf_string_to_oemol(sdf_str: str) -> oechem.OEGraphMol:
    """
    Loads an SDF string into an openeye molecule

    Parameters
    ----------
    sdf_str: str
       The string representation of an SDF file

    Returns
    -------
    oechem.OEMol:
       resulting OpenEye OEMol
    """

    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SDF)
    ims.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    ims.openstring(sdf_str)
    # NOTE: must use GraphMol here, not OEMol, otherwise SD data will not be read
    mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ims, mol)
    return mol


def smiles_to_oemol(smiles: str) -> oechem.OEGraphMol:
    """
    Loads a smiles string into an openeye molecule

    Parameters
    ----------
    smiles: str
       SMILES string

    Returns
    -------
    oechem.OEGraphMol
        resulting OpenEye OEGraphMol
    """
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    return mol


def oemol_to_smiles(mol: oechem.OEMol) -> str:
    """
    SMILES string of an OpenEye OEMol

    Paramers
    --------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    str
       SMILES string of molecule
    """
    return oechem.OEMolToSmiles(mol)


def oemol_to_inchi(mol: oechem.OEMol) -> str:
    """
    InChI string of an OpenEye OEMol

    Paramers
    --------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    str
       InChI string of molecule
    """
    return oechem.OECreateInChI(mol)


def oemol_to_inchikey(mol: oechem.OEMol) -> str:
    """
    InChI key string of an OpenEye OEMol

    Paramers
    --------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    str
       InChI key string of molecule
    """
    return oechem.OECreateInChIKey(mol)


def set_SD_data(mol: oechem.OEMol, key: str, value: str) -> oechem.OEMol:
    """
    Set the SD data on an OpenEye OEMol

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    oechem.OEMol
        OpenEye OEMol with SD data set
    """
    try:
        key = str(key)
        value = str(value)
    except:
        raise Exception(f"SD data key {key} or value {value} is not castable  a string")
    oechem.OESetSDData(mol, key, value)
    return mol


def set_SD_data_dict(mol: oechem.OEMol, data: Dict[str, str]) -> oechem.OEMol:
    """
    Set the SD data on an OpenEye OEMol, overwriting any existing data with the same tag

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    oechem.OEMol
        OpenEye OEMol with SD data set
    """
    for key, value in data.items():
        try:
            key = str(key)
            value = str(value)
        except:
            raise Exception(
                f"SD data key {key} or value {value} is not castable  a string"
            )
        oechem.OESetSDData(mol, key, value)
    return mol


def get_SD_data(mol: oechem.OEMol, key: str) -> str:
    """
    Get the SD data on an OpenEye OEMol

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    str
        SD data value
    """
    return oechem.OEGetSDData(mol, key)


def get_SD_data_dict(mol: oechem.OEMol) -> Dict[str, str]:
    """
    Get all SD data on an OpenEye OEMol

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    Dict[str, str]
        Dictionary of SD data
    """
    sd_data = {}
    for dp in oechem.OEGetSDDataPairs(mol):
        sd_data[dp.GetTag()] = dp.GetValue()
    return sd_data


def print_SD_Data(mol: oechem.OEMol) -> None:
    print("SD data of", mol.GetTitle())
    # loop over SD data
    for dp in oechem.OEGetSDDataPairs(mol):
        print(dp.GetTag(), ":", dp.GetValue())
