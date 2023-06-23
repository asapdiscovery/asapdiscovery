from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from asapdiscovery.data.openeye import (
    oechem,
    oemol_to_sdf_string,
    oemol_to_smiles,
    sdf_string_to_oemol,
    smiles_to_oemol,
)
from asapdiscovery.data.schema import ExperimentalCompoundData
from pydantic import BaseModel, ByteSize, Field, validator


class InvalidLigandError(ValueError):
    ...


class DataStorageType(str, Enum):
    sdf = "sdf"
    pdb = "pdb"


def read_file_directly(file: str | Path) -> str:
    with open(str(file)) as f:
        contents = f.read()
    return contents


def write_file_directly(file: str | Path, data: str) -> None:
    with open(str(file), "w") as f:
        f.write(data)


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
    postera_vc_id: str | None = Field(None, description="Unique VC ID from Postera")


class Ligand(DataModelAbstractBase):
    """
    Schema for a Ligand
    """

    compound_name: str = Field(None, description="Name of compound")
    ids: LigandIdentifiers | None = Field(
        None,
        description="LigandIdentifiers Schema for identifiers associated with this ligand",
    )
    experimental_data: ExperimentalCompoundData | None = Field(
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
        sdf_str = oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        mol = sdf_string_to_oemol(self.data)
        return mol

    @classmethod
    def from_smiles(cls, smiles: str, compound_name: str = None, **kwargs) -> Ligand:
        mol = smiles_to_oemol(smiles)
        sdf_str = oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    @property
    def smiles(self) -> str:
        mol = sdf_string_to_oemol(self.data)
        return oemol_to_smiles(mol)

    @classmethod
    def from_sdf(
        cls, sdf_file: str | Path, compound_name: str = None, **kwargs
    ) -> Ligand:
        # directly read in data
        sdf_str = read_file_directly(sdf_file)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_sdf(self, filename: str | Path) -> None:
        # directly write out data
        write_file_directly(filename, self.data)


class ReferenceLigand(Ligand):
    target_name: str | None = None
