from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ByteSize, ConfigDict, Field

_SCHEMA_VERSION = "0.1.0"


class DataStorageType(str, Enum):
    sdf = "sdf"
    pdb = "pdb"
    b64oedu = "b64oedu"


def read_file_directly(file: str | Path) -> str:
    with open(str(file)) as f:
        contents = f.read()
    return contents


def write_file_directly(file: str | Path, data: str, mode: str = "w") -> None:
    if mode not in ["w", "a"]:
        raise ValueError(f"mode must be either 'w' or 'a', got {mode}")
    with open(str(file), mode) as f:
        f.write(data)


def utf8len(s: str) -> int:
    return len(s.encode("utf-8"))


def check_strings_for_equality_with_exclusion(string1, string2, exclusion_string):
    lines1 = [line for line in string1.split("\n") if exclusion_string not in line]
    lines2 = [line for line in string2.split("\n") if exclusion_string not in line]
    return lines1 == lines2


class DataModelAbstractBase(BaseModel):
    """
    Base class for asapdiscovery pydantic models that simplify dictionary, JSON
    and other behaviour
    """

    # can't use extra="forbid" because of the way we use
    # kwargs to skip root_validator on some fields
    model_config = ConfigDict(validate_assignment=True)

    def __hash__(self) -> int:
        return self.model_dump_json().__hash__()

    @classmethod
    def from_dict(cls, dict):
        return cls.model_validate(dict)

    @classmethod
    def from_json(cls, json_str):
        return cls.model_validate(json.loads(json_str))

    @classmethod
    def from_json_file(cls, file: str | Path):
        with open(str(file)) as f:
            return cls.model_validate_json(f.read())

    def to_json_file(self, file: str | Path):
        write_file_directly(file, self.model_dump_json())

    @property
    def size(self) -> ByteSize:
        """Size of the resulting JSON object for this class"""
        return ByteSize(utf8len(self.model_dump_json())).human_readable()

    def full_equal(self, other: DataModelAbstractBase) -> bool:
        return self.model_dump() == other.model_dump()

    def data_equal(self, other: DataModelAbstractBase) -> bool:
        return self.data == other.data

    def get_schema_version(self) -> str:
        return _SCHEMA_VERSION

    # use data_equal instead
    def __eq__(self, other: DataModelAbstractBase) -> bool:
        # check if has a data attribute
        if hasattr(self, "data"):
            return self.data_equal(other)
        else:
            return self.full_equal(other)

    # use data_equal instead
    def __ne__(self, other: DataModelAbstractBase) -> bool:
        return not self.__eq__(other)


def schema_dict_get_val_overload(obj: dict | BaseModel):
    """
    Overload for Schema and Dict to get values easily

    Parameters
    ----------
    obj : Union[Dict, Schema]
        Object to get values from

    Returns
    -------
    Iterable[Any]
    """
    if isinstance(obj, dict):
        return obj.values()
    elif isinstance(obj, BaseModel):
        return obj.model_dump().values()
    else:
        raise TypeError(f"Unsupported type {type(obj)}")


class ComplexBase(DataModelAbstractBase):
    """
    Base class for complexes
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComplexBase):
            return NotImplemented

        # Just check that both Targets and Ligands are the same
        return (self.target == other.target) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def unique_name(self) -> str:
        """Create a unique name for the Complex, this is used in prep when generating folders to store results."""
        return f"{self.target.target_name}-{self.hash}"


class MoleculeComponent(str, Enum):
    PROTEIN = "protein"
    LIGAND = "ligand"
    WATER = "water"
    OTHER = "other"


class MoleculeFilter(BaseModel):
    """Filter for selecting components of a molecule."""

    model_config = ConfigDict(extra="forbid")

    protein_chains: list = Field(
        default_factory=list,
        description="List of chains containing the desired protein. An empty list will return all chains.",
    )
    ligand_chain: str | None = Field(
        None,
        description="Chain containing the desired ligand. An empty list will return all chains.",
    )
    water_chains: list = Field(
        default_factory=list,
        description="List of chains containing the desired water. An empty list will return all chains.",
    )
    other_chains: list = Field(
        default_factory=list,
        description="List of chains containing other items. An empty list will return all chains.",
    )
    components_to_keep: list[MoleculeComponent] = Field(
        default_factory=lambda: ["protein", "ligand", "water", "other"],
        description="List of components to keep. An empty list will return all components.",
    )
