from __future__ import annotations

from pydantic import BaseModel, Field, validator, ByteSize

from asapdiscovery.data.openeye import (
    oechem,
    oespruce,
    load_openeye_pdb,
    save_openeye_pdb,
    load_openeye_sdf,
)
from asapdiscovery.modeling.modeling import make_design_unit
from asapdiscovery.data.schema import ExperimentalCompoundData

from enum import Enum
import json
from typing import Dict, Union, Optional, Any, Tuple
from pathlib import Path



class InvalidLigandError(ValueError):
    ...


class DataStorageType(str, Enum):
    sdf = "sdf"
    pdb = "pdb"


def _oemol_to_sdf_string(mol: oechem.OEMol) -> str:
    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_SDF)
    oms.openstring()
    oechem.OEWriteMolecule(oms, mol)
    molstring = oms.GetString().decode("UTF-8")
    return molstring


def _sdf_string_to_oemol(sdf_str: str) -> oechem.OEMol:
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SDF)
    ims.openstring(sdf_str)
    mols = []
    mol = oechem.OEMol()
    for mol in ims.GetOEMols():
        mols.append(oechem.OEMol(mol))
    if len(mols) != 1:
        raise InvalidLigandError("more than one molecule in input stream")
    return mols[0]


def _read_file_directly(file: Union[str, Path]) -> str:
    with open(str(file), "r") as f:
        contents = f.read()
    return contents


def _write_file_directly(file: Union[str, Path], data: str) -> None:
    with open(str(file), "w") as f:
        f.write(data)


def _smiles_to_oemol(smiles) -> oechem.OEGraphMol:
    # Create an empty molecule object
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    return mol


def _oemol_to_smiles(mol: oechem.OEMol) -> str:
    return oechem.OEMolToSmiles(mol)


def utf8len(s: str) -> int:
    return len(s.encode("utf-8"))


class DataModelAbstractBase(BaseModel):
    """
    Base class for asapdiscovery pydantic models that simplify dictionary, JSON
    and other behaviour
    """

    @classmethod
    def from_dict(cls, dict):
        return cls.parse_obj(dict)

    @classmethod
    def from_json(cls, json_str):
        return cls.parse_obj(json.loads(json_str))

    # check_fields required to be false as `data` not defined for baseclass
    @validator("data", check_fields=False)
    def data_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Data field can not be empty")
        return v

    @property
    def size(self) -> ByteSize:
        """Size of the resulting JSON object for this class"""
        return ByteSize(utf8len(self.json())).human_readable()

    def data_equal(self, other: DataModelAbstractBase) -> Bool:
        return self.data == other.data

    class Config:
        validate_assignment = True


class LigandIdentifiers(DataModelAbstractBase):
    """
    Identifiers for a Ligand
    """

    moonshot_compound_id: str = Field(None, description="Moonshot compound ID")
    postera_vc_id: Optional[str] = Field(None, description="Unique VC ID from Postera")


class Ligand(DataModelAbstractBase):
    """
    Schema for a Ligand
    """

    compound_name: str = Field(None, description="Name of compound")
    ids: Optional[LigandIdentifiers] = Field(
        None,
        description="LigandIdentifiers Schema for identifiers associated with this ligand",
    )
    experimental_data: Optional[ExperimentalCompoundData] = Field(
        None,
        description="ExperimentalCompoundData Schema for experimental data associated with the compound",
    )
    data: str = Field(
        "",
        description="SDF file stored as a string to hold internal data state",
        repr=False,
    )
    data_format: DataStorageType = Field(
        DataStorageType.sdf,
        description="Enum describing the data storage method",
        allow_mutation=False,
    )

    @classmethod
    def from_oemol(
        cls, mol: oechem.OEMol, compound_name: str = None, **kwargs
    ) -> Ligand:
        sdf_str = _oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        mol = _sdf_string_to_oemol(self.data)
        return mol

    @classmethod
    def from_smiles(cls, smiles: str, compound_name: str = None, **kwargs) -> Ligand:
        mol = _smiles_to_oemol(smiles)
        sdf_str = _oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    @property
    def smiles(self) -> str:
        mol = _sdf_string_to_oemol(self.data)
        return _oemol_to_smiles(mol)

    @classmethod
    def from_sdf(
        cls, sdf_file: Union[str, Path], compound_name: str = None, **kwargs
    ) -> Ligand:
        # directly read in data
        sdf_str = _read_file_directly(sdf_file)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_sdf(self, filename: Union[str, Path]) -> None:
        # directly write out data
        _write_file_directly(filename, self.data)


class ReferenceLigand(Ligand):
    target_name: Optional[str] = None