from pathlib import Path
from typing import Union

from asapdiscovery.data.schema.schema_base import read_file_directly
from rdkit import Chem


def _set_SD_data(mol: Union[Chem.Mol, Chem.Conformer], data: dict[str, str]):
    """
    Set the SD data on an rdkit molecule or conformer

    Parameters
    ----------
    mol: Union[Chem.Mol, Chem.Conformer]
        rdkit molecule or conformer

    data: dict[str, str]
        Dictionary of SD data to set
    """
    for key, value in data.items():
        mol.SetProp(str(key), str(value))


def set_SD_data(mol: Chem.Mol, data: dict[str, str | list]):
    """
    Set the SD data on an rdkit molecule, overwriting any existing data.
    If the length of a list is 1, will set that value to all conformers.
    If the length of a list is equal to the number of conformers, will set each value to the corresponding conformer.
    Finally, it will set the properties for the whole molecule to be the data for the first conformer.
    Otherwise, will raise a ValueError.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        rdkit molecule

    data: dict[str, list]
        Dictionary of SD data to set.
        Each key should be a tag name and each value should be a list of values, one for each conformer.
    """
    num_confs = mol.GetNumConformers()

    # convert to dict of lists first
    data = {k: v if isinstance(v, list) else [v] for k, v in data.items()}

    for key, value in data.items():
        if len(value) == 1:
            for conf in mol.GetConformers():
                conf.SetProp(str(key), str(value[0]))
        elif len(value) == num_confs:
            for i, conf in enumerate(mol.GetConformers()):
                conf.SetProp(str(key), str(value[i]))
        else:
            raise ValueError(
                f"Length of data for tag '{key}' does not match number of conformers ({num_confs}). "
                f"Expected {num_confs} but got {len(value)} elements."
            )

    # Set the properties for the highest level to be the data for the first conformer
    from asapdiscovery.data.util.data_conversion import get_first_value_of_dict_of_lists

    first_conf_data = get_first_value_of_dict_of_lists(data)
    _set_SD_data(mol, first_conf_data)


def _get_SD_data(mol: Union[Chem.Mol, Chem.Conformer]) -> dict[str, str]:
    """
    Get the SD data from an RDKit molecule or conformer

    Parameters
    ----------
    mol: Union[Chem.Mol, Chem.Conformer]
        RDKit molecule or conformer

    Returns
    -------
    dict
        Dictionary of SD data
    """
    return mol.GetPropsAsDict()


def get_SD_data(mol: Chem.Mol) -> dict[str, list]:
    """
    Get the SD data from an RDKit molecule.
    If there are multiple conformers, will get data from the conformers,
    so properties saved to mol.Prop will be ignored.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule

    Returns
    -------
    dict
        Dictionary of SD data
    """
    if mol.GetNumConformers() == 1:
        from asapdiscovery.data.util.data_conversion import (
            get_dict_of_lists_from_dict_of_str,
        )

        return get_dict_of_lists_from_dict_of_str(_get_SD_data(mol))

    from asapdiscovery.data.util.data_conversion import (
        get_dict_of_lists_from_list_of_dicts,
    )

    data_list = [_get_SD_data(conf) for conf in mol.GetConformers()]
    return get_dict_of_lists_from_list_of_dicts(data_list)


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
    suppl = Chem.ForwardSDMolSupplier(bio, removeHs=False)

    ref = next(suppl)
    for mol in suppl:
        data = mol.GetPropsAsDict()
        conf = mol.GetConformer()
        _set_SD_data(conf, data)
        ref.AddConformer(conf, assignId=True)
    return ref


def rdkit_mol_to_sdf_str(mol: Chem.Mol) -> str:
    """
    Convert an RDKit molecule to a SDF string

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule

    Returns
    -------
    str
        SDF string
    """
    from io import StringIO

    sdfio = StringIO()
    w = Chem.SDWriter(sdfio)
    w.write(mol)
    w.flush()
    return sdfio.getvalue()


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
