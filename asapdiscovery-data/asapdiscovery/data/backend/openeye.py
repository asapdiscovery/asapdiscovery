from base64 import b64decode, b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional, Union  # noqa: F401
from warnings import warn

from openeye import (  # noqa: F401
    oechem,
    oedepict,
    oedocking,
    oeff,
    oegrid,
    oeomega,
    oequacpac,
    oeshape,
    oespruce,
    oeszybki,
)

# exec on module import

if not oechem.OEChemIsLicensed("python"):
    warn("OpenEye license required to use asapdiscovery openeye module")


def combine_protein_ligand(
    prot: oechem.OEMol,
    lig: oechem.OEMol,
    lig_name: str = "LIG",
    lig_chain: Optional[str] = "",
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
        if lig_chain:
            res.SetChainID(lig_chain)
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


def load_openeye_smi(smi_fn: Union[str, Path]) -> list[oechem.OEGraphMol]:
    """
    Load an OpenEye SMILES file containing a set of molecules and return them as
    OpenEye OEGraphMol objects.
    Parameters
    ----------
    smi_fn : Union[str, Path]
        Path to the SMILES file to load.
    Returns
    -------
    list[oechem.OEGraphMol]
        A list of OpenEye OEGraphMol objects corresponding to the data from the SMI file.
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    oechem.OEError
        If the SMI file cannot be opened.
    """
    # convert to path to make consistent
    smi_fn = Path(smi_fn)
    if not smi_fn.exists():
        raise FileNotFoundError(f"{str(smi_fn)} does not exist!")

    ifs = oechem.oemolistream(smi_fn.as_posix())
    ifs.SetFlavor(oechem.OEFormat_SMI, oechem.OEIFlavor_SMI_DEFAULT)

    molecules = []
    for mol in ifs.GetOEGraphMols():
        molecules.append(oechem.OEGraphMol(mol))

    return molecules


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


def load_openeye_sdf(sdf_fn: Union[str, Path]) -> oechem.OEMol:
    """
    Load an OpenEye SDF file and return it as an OpenEye OEMol object.
    Reads multiple conformers into the OEMol object but if the sdf file contains
    multiple molecules, it will only return the first one.

    Parameters
    ----------
    sdf_fn : Union[str, Path]
        Path to the SDF file to load.

    Returns
    -------
    oechem.OEMol
        An OpenEye OEMol object containing the molecule data from the SDF file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    oechem.OEError
        If the SDF file cannot be opened.

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
    ifs.SetConfTest(oechem.OEOmegaConfTest())
    if ifs.open(str(sdf_fn)):
        for mol in ifs.GetOEMols():
            ifs.close()
            return mol
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


def save_openeye_design_unit(du: oechem.OEDesignUnit, du_fn: Union[str, Path]) -> Path:
    """
    Write an OpenEye design unit to a file

    Parameters
    ----------
    du : oechem.OEDesignUnit
        The OpenEye DesignUnit to write to the file.
    du_fn : Union[str, Path]
        The path of the DesignUnit file to create or overwrite.

    Returns
    -------
    Path
        The path of the DesignUnit file that was written.

    Raises
    ------
    oechem.OEError
        If the DesignUnit file cannot be opened.

    Notes
    -----
    This function will overwrite any existing file with the same name as `du_fn`.
    """
    retcode = oechem.OEWriteDesignUnit(str(du_fn), du)
    if not retcode:
        oechem.OEThrow.Fatal(f"Unable to open {du_fn}")
    return Path(du_fn)


def openeye_perceive_residues(
    prot: oechem.OEGraphMol, preserve_all: bool = False
) -> oechem.OEGraphMol:
    """
    Re-perceive the residues of a protein molecule using OpenEye's OEPerceiveResidues function,
    which is necessary when changes are made to the protein to ensure correct atom ordering and CONECT record creation.

    Parameters
    ----------
    prot : oechem.OEGraphMol
        The input protein molecule to be processed.

    preserve_all : bool, optional, default=False
        If True, preserve all residue information, including chain ID, residue number, and residue name.

    Returns
    -------
    oechem.OEGraphMol
        The processed protein molecule with re-perceived residue information.
    """
    # Clean up PDB info by re-perceiving, perserving chain ID, residue number, and residue name

    if preserve_all:
        preserve = oechem.OEPreserveResInfo_All
    else:
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
    Path
        Path to the receptor grid file
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


def sdf_string_to_oemol(sdf_str: str) -> oechem.OEMol:
    """
    Loads an SDF string into an openeye molecule.
    Enables multiple conformers but only returns the first molecule in a multi molecule sdf

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
    rmol = oechem.OEMol()
    ims.SetConfTest(oechem.OEOmegaConfTest())
    if ims.openstring(sdf_str):
        for mol in ims.GetOEMols():
            rmol = mol.CreateCopy()
            break  # only return the first molecule
    return rmol


def smiles_to_oemol(smiles: str) -> oechem.OEMol:
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
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    return mol


def oemol_to_smiles(mol: oechem.OEMol, isomeric=True) -> str:
    """
    Canonical SMILES string of an OpenEye OEMol

    Paramers
    --------
    mol: oechem.OEMol
        OpenEye OEMol

    isomeric: bool, optional, default=True
        If True, generate canonical isomeric SMILES (including stereochem)
        If False, generate canonical SMILES without stereochem

    Returns
    -------
    str
       SMILES string of molecule
    """
    # By default, OEMolToSmiles generates isomeric SMILES, which includes stereochemistry
    if isomeric:
        return oechem.OEMolToSmiles(mol)

    # However, if we want to treat two stereoisomers as the same molecule,
    # we can generate canonical SMILES that don't include stereo info
    else:
        return oechem.OECreateCanSmiString(mol)


def oe_smiles_roundtrip(smiles: str) -> str:
    """
    Canonical SMILES string of an OpenEye OEMol

    Paramers
    --------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    str
       SMILES string of molecule
    """
    mol = smiles_to_oemol(smiles)
    return oemol_to_smiles(mol)


def oemol_to_inchi(mol: oechem.OEMol, fixed_hydrogens: bool = False) -> str:
    """
    InChI string of an OpenEye OEMol

    Paramers
    --------
    mol: oechem.OEMol
        OpenEye OEMol
    fixed_hydrogens: bool
        If a fixed hydrogen layer should be added to the InChI, if `True` this will result in a non-standard inchi
        which can distinguish tautomers.

    Returns
    -------
    str
       InChI string of molecule
    """
    if fixed_hydrogens:
        inchi_opts = oechem.OEInChIOptions()
        inchi_opts.SetFixedHLayer(True)
        inchi = oechem.OEMolToInChI(mol)
    else:
        inchi = oechem.OEMolToSTDInChI(mol)

    return inchi


def oemol_to_inchikey(mol: oechem.OEMol, fixed_hydrogens: bool = False) -> str:
    """
    InChI key string of an OpenEye OEMol

    Paramers
    --------
    mol: oechem.OEMol
        OpenEye OEMol

    fixed_hydrogens: bool
        If a fixed hydrogen layer should be added to the InChI, if `True` this will result in a non-standard inchi
        which can distinguish tautomers.
    Returns
    -------
    str
       InChI key string of molecule
    """
    if fixed_hydrogens:
        inchi_opts = oechem.OEInChIOptions()
        inchi_opts.SetFixedHLayer(True)
        inchi_key = oechem.OEMolToInChIKey(mol)
    else:
        inchi_key = oechem.OEMolToSTDInChIKey(mol)

    return inchi_key


def _set_SD_data(mol: oechem.OEMolBase, data: dict[str, str]) -> oechem.OEMolBase:
    """
    Set SD data on an OpenEye OEMolBase object.
    Since this function works on OEMol, OEGraphMol, OEConfBase objects, it is worth repurposing.
    But it is not recommended to use this function directly for multi-conformer molecules.

    Parameters
    ----------
    mol: oechem.OEMolBase
        OpenEye OEMolBase

    Returns
    -------
    oechem.OEMolBase
        OpenEye OEMolBase with SD data set
    """
    for key, value in data.items():
        oechem.OESetSDData(mol, key, str(value))
    return mol


def set_SD_data(mol: oechem.OEMol, data: dict[str, str | list]) -> oechem.OEMol:
    """
    Set the SD data on an OpenEye OEMol, overwriting any existing data with the same tag
    If a str or a single-length list is passed as the values of the dictionary, the data will be set to all conformers.
    If a list is provided, the data will be set to the conformers in the order provided.
    If the list is not the same length as the number of conformers, an error will be raised.

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    data: dict[str, str | list]
        Dictionary of SD data to set.


    Returns
    -------
    oechem.OEMol
        OpenEye OEMol with SD data set
    """
    from asapdiscovery.data.util.data_conversion import get_first_value_of_dict_of_lists

    # convert to dict of lists first
    data = {k: v if isinstance(v, list) else [v] for k, v in data.items()}

    # If the object is an OEMol, we will set the SD data to all the conformers
    if isinstance(mol, oechem.OEMol):
        for key, value_list in data.items():
            # if list is len 1, generate a list of len N, where N is the number of conformers
            if len(value_list) == 1:
                value_list = value_list * mol.NumConfs()

            if not len(value_list) == mol.NumConfs():
                raise ValueError(
                    f"Length of data for tag '{key}' does not match number of conformers ({mol.NumConfs()}). "
                    f"Expected {mol.NumConfs()} but got {len(value_list)} elements."
                )
            for i, conf in enumerate(mol.GetConfs()):
                oechem.OESetSDData(conf, key, str(value_list[i]))
        return mol
    elif isinstance(mol, oechem.OEMolBase):
        return _set_SD_data(mol, get_first_value_of_dict_of_lists(data))

    else:
        raise TypeError(
            f"Expected an OpenEye OEMol, OEGraphMol, or OEConf, but got {type(mol)}"
        )


def _get_SD_data(mol: oechem.OEMolBase) -> dict[str, str]:
    """
    Get SD data from an OpenEye OEMolBase object.
    Since this function works on OEMol, OEGraphMol, OEConfBase objects, it is worth repurposing.
    But it is not recommended to use this function directly for multi-conformer molecules.

    Parameters
    ----------
    mol: oechem.OEMolBase
        OpenEye OEMolBase

    Returns
    -------
    Dict[str, str]
        Dictionary of SD data
    """
    return {dp.GetTag(): dp.GetValue() for dp in oechem.OEGetSDDataPairs(mol)}


def get_SD_data(mol: oechem.OEMolBase) -> dict[str, list]:
    """
    Get all SD data on an OpenEye OEMol, OEGraphMol, or OEConfBase object.
    If multiple conformers are found, the SD tags from the conformers will be used, and any properties at the highest
    level will be ignored.

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    dict[str, list]
        Dictionary of SD data

    Raises
    ------
    TypeError
        If mol is a type that cant be converted to an OEMol, OEGraphMol, or OEConfBase
    """
    from asapdiscovery.data.util.data_conversion import (
        get_dict_of_lists_from_dict_of_str,
        get_dict_of_lists_from_list_of_dicts,
    )

    # If the object is an OEMol, we have to pull from the conformers, because even if there is only one conformer
    # the data is stored at the conformer level if you generate an oemol from an sdf file
    # However, if you've manually fiddled with the tags, the data might be stored at the molecule level.
    # In order to resolve this, if there is data at the high level that is not repeated at the conformer level,
    # we'll add to all the conformers and return that dict of lists.

    if isinstance(mol, oechem.OEMol):
        # Get the data from the molecule
        molecule_tags = _get_SD_data(mol)

        # get the data from the conformers
        conformer_tags = get_dict_of_lists_from_list_of_dicts(
            [_get_SD_data(conf) for conf in mol.GetConfs()]
        )

        for k, v in molecule_tags.items():
            if k not in conformer_tags:
                conformer_tags[k] = [v] * mol.NumConfs()

        return conformer_tags
    elif isinstance(mol, oechem.OEMolBase):
        return get_dict_of_lists_from_dict_of_str(_get_SD_data(mol))
    else:
        raise TypeError(
            f"Expected an OpenEye OEMol, OEGraphMol, or OEConf, but got {type(mol)}"
        )


def _get_SD_data_to_object(mol: oechem.OEMol) -> dict[str, Any]:
    """
    Get all SD data on an OpenEye OEMol, converting to Python objects

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    Dict[str, Any]
        Dictionary of SD data
    """
    import ast

    sd_data = get_SD_data(mol)
    for key, value in sd_data.items():
        sd_data[key] = ast.literal_eval(value)
    return sd_data


def print_SD_data(mol: oechem.OEMol) -> None:
    print("SD data of", mol.GetTitle())
    # loop over SD data
    for dp in oechem.OEGetSDDataPairs(mol):
        print(dp.GetTag(), ":", dp.GetValue())


def clear_SD_data(mol: oechem.OEMol) -> oechem.OEMol:
    """
    Clear all SD data on an OpenEye OEMol

    Parameters
    ----------
    mol: oechem.OEMol
        OpenEye OEMol

    Returns
    -------
    oechem.OEMol
        OpenEye OEMol with SD data cleared
    """
    for conf in mol.GetConfs():
        oechem.OEClearSDData(conf)
    return mol


def oemol_to_pdb_string(mol: oechem.OEMol) -> str:
    """
    Dumps an OpenEye OEMol to a PDB string

    Parameters
    ----------
    mol: oechem.OEMol
         OpenEye OEMol

    Returns
    -------
    str
        PDB string representation of the input OEMol
    """
    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_PDB)
    oms.SetFlavor(oechem.OEFormat_PDB, oechem.OEOFlavor_PDB_Default)
    oms.openstring()
    oechem.OEWriteMolecule(oms, mol)
    molstring = oms.GetString().decode("UTF-8")
    return molstring


def pdb_string_to_oemol(pdb_str: str) -> oechem.OEGraphMol:
    """
    Loads a PDB string into an OpenEye OEGraphMol

    Parameters
    ----------
    pdb_str: str
        The string representation of a PDB file

    Returns
    -------
    oechem.OEMol
        resulting OpenEye OEMol
    """
    ifs = oechem.oemolistream()
    ifs.SetFormat(oechem.OEFormat_PDB)
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        oechem.OEIFlavor_PDB_Default
        | oechem.OEIFlavor_PDB_DATA
        | oechem.OEIFlavor_PDB_ALTLOC,
    )  # noqa
    ifs.openstring(pdb_str)
    mol = oechem.OEGraphMol()
    if not oechem.OEReadMolecule(ifs, mol):
        oechem.OEThrow.Fatal("Cannot read molecule")
    return mol


def oedu_to_bytes64(oedu: oechem.OEDesignUnit) -> bytes:
    """
    Convert an OpenEye DesignUnit to bytes

    Parameters
    ----------
    oedu: oechem.OEDesignUnit
        OpenEye DesignUnit

    Returns
    -------
    bytes
        bytes representation of the input DesignUnit encoded in base64
    """
    oedu_bytes = oechem.OEWriteDesignUnitToBytes(oedu)
    # convert to base64
    return b64encode(oedu_bytes)


def bytes64_to_oedu(bytes: bytes) -> oechem.OEDesignUnit:
    """
    Convert bytes to an OpenEye DesignUnit

    Parameters
    ----------
    bytes: bytes
        bytes representation of a DesignUnit encoded in base64

    Returns
    -------
    oechem.OEDesignUnit
        resulting OpenEye DesignUnit
    """
    # convert from base64
    bytes = b64decode(bytes)
    du = oechem.OEDesignUnit()
    retcode = oechem.OEReadDesignUnitFromBytes(du, bytes)
    if not retcode:
        oechem.OEThrow.Fatal("Cannot read DesignUnit from bytes")
    return du


def featurize_oemol(mol: oechem.OEMol, self_edges=True):
    """
    Featurize an OE molecule for use in ML. Returns a feature tensor of shape
    (n_atoms, n_features) and an edge index tensor, which will have one of two shapes:
        * (2, 2 * n_bonds) if self_edges==False
        * (2, 2 * n_bonds + n_atoms) otherwise
    Each bond gives 2 edges because each atom can be a source or destination node.

    The featurization scheme closely matches that of DGL-LifeSci, with the main
    difference being the size of the atom type one-hot encoding.


    Parameters
    ----------
    mol: oechem.OEMol
        Molecule to featurize
    self_edges: bool, default=True
        Should we include edges going from each atom to itself

    Returns
    -------
    torch.Tensor
        Feature tensor of shape (n_atom, n_features)
    torch.Tensor
        Edge index tensor of shape (2, 2 * n_bonds) or (2, 2 * n_bonds + n_atoms)
    """
    import torch

    # Make a copy of the molecule so we don't modify the original
    mol = mol.CreateCopy()

    # Theoretically any GNN that you use should be permutation-invariant, but just for
    #  some semblance of determinism
    oechem.OECanonicalOrderAtoms(mol)
    oechem.OECanonicalOrderBonds(mol)

    # Gather all relevant info
    atom_info = [
        (
            a.GetAtomicNum(),
            a.GetDegree(),
            a.GetImplicitHCount(),
            a.GetFormalCharge(),
            a.GetHyb(),
            a.IsAromatic(),
            a.GetTotalHCount(),
        )
        for a in mol.GetAtoms()
    ]

    # Split out into individual lists
    (
        atom_types,
        atom_degrees,
        atom_implicit_h,
        atom_formal_charges,
        atom_hyb,
        atom_aromatic,
        atom_total_h,
    ) = zip(*atom_info)

    # Atom type one-hot encoding
    atom_types_one_hot = torch.nn.functional.one_hot(
        torch.tensor(atom_types), num_classes=oechem.OEElemNo_MAXELEM
    )

    # Atom degrees
    atom_degrees_one_hot = torch.nn.functional.one_hot(
        torch.tensor(atom_degrees), 11
    )  # 11 comes from dgl-lifesci

    # Implicit Hs
    atom_implicit_h_one_hot = torch.nn.functional.one_hot(
        torch.tensor(atom_implicit_h), 7
    )  # 7 comes from dgl-lifesci

    # Formal charges
    atom_formal_charges = torch.tensor(atom_formal_charges).reshape((-1, 1))

    # Hybridization states
    atom_hyb_one_hot = torch.nn.functional.one_hot(
        torch.tensor(atom_hyb), 6
    )  # 6 comes from number of hybridization states in OE

    # Aromaticity
    atom_aromatic = torch.tensor(atom_aromatic).reshape((-1, 1))

    # Total Hs
    atom_total_h_one_hot = torch.nn.functional.one_hot(
        torch.tensor(atom_total_h), 5
    )  # 7 comes from dgl-lifesci

    # Combine all features
    feature_tensor = torch.hstack(
        [
            atom_types_one_hot,
            atom_degrees_one_hot,
            atom_implicit_h_one_hot,
            atom_formal_charges,
            atom_hyb_one_hot,
            atom_aromatic,
            atom_total_h_one_hot,
        ]
    ).to(dtype=torch.float)

    # Bonds (make sure we get both directions)
    bond_list = [
        atom_pair
        for bond in mol.GetBonds()
        for atom_pair in (
            (bond.GetBgnIdx(), bond.GetEndIdx()),
            (bond.GetEndIdx(), bond.GetBgnIdx()),
        )
    ]

    # Add in self-edges if desired
    bond_list += [(a.GetIdx(), a.GetIdx()) for a in mol.GetAtoms()]

    # Cast bonds to tensor and transpose so order is correct
    bond_list_tensor = torch.tensor(bond_list).T

    return feature_tensor, bond_list_tensor
