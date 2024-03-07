from rdkit import Chem
from typing import Union
from pathlib import Path
from asapdiscovery.data.schema.schema_base import read_file_directly


def set_SD_data(mol: Chem.Conformer, data: dict) -> None:
    for key, value in data.items():
        mol.SetProp(str(key), str(value))


def set_multiconf_SD_data(mol: Chem.Mol, data: dict[str, list]):
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
    num_confs = mol.GetNumConformers()
    for key, value in data.items():
        if not len(value) == num_confs:
            raise ValueError(
                f"Length of data for tag '{key}' does not match number of conformers ({num_confs}). "
                f"Expected {num_confs} but got {len(value)} elements."
            )
        for i, conf in enumerate(mol.GetConformers()):
            set_SD_data(conf, data)


def get_multiconf_SD_data(mol: Chem.Mol) -> dict[str, list]:
    """
    Get the SD data from an RDKit molecule

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule

    Returns
    -------
    dict
        Dictionary of SD data
    """
    data = {}
    for conf in mol.GetConformers():
        conf_data = conf.GetPropsAsDict()
        for key, value in conf_data.items():
            if key not in data:
                data[key] = []
            data[key].append(value)
    return data


def load_sdf(file: Union[str, Path]) -> Chem.Mol:
    """
    Load an SDF file into an RDKit molecule
    """
    sdf_str = read_file_directly(file)
    return sdf_str_to_rdkit_mol(sdf_str)


def sdf_str_to_rdkit_mol(sdf: str) -> Chem.Mol:
    """
    Convert a SDF string to an RDKit molecule

    Parameters
    ----------
    sdf : str
        SDF string

    Returns
    -------
    Chem.Mol
        RDKit molecule
    """
    from io import BytesIO

    bio = BytesIO(sdf.encode())
    suppl = Chem.ForwardSDMolSupplier(bio)

    ref = next(suppl)
    for mol in suppl:
        data = mol.GetPropsAsDict()
        conf = mol.GetConformer()
        set_SD_data(conf, data)
        ref.AddConformer(conf, assignId=True)
    return ref


def rdkit_smiles_roundtrip(smi: str) -> str:
    """
    Roundtrip a SMILES string through RDKit to canonicalize it

    Parameters
    ----------
    smi : str
        SMILES string to canonicalize

    Returns
    -------
    str
        Canonicalized SMILES string
    """
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol)
