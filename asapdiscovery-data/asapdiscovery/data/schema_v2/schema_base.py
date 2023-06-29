from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from pydantic import BaseModel, ByteSize


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

    @property
    def size(self) -> ByteSize:
        """Size of the resulting JSON object for this class"""
        return ByteSize(utf8len(self.json())).human_readable()

    def data_equal(self, other: DataModelAbstractBase) -> bool:
        return self.data == other.data

    # use data_equal instead
    def __eq__(self, other: DataModelAbstractBase) -> bool:
        return self.data_equal(other)

    # use data_equal instead
    def __ne__(self, other: DataModelAbstractBase) -> bool:
        return not self.data_equal(other)

    class Config:
        validate_assignment = True
        # can't use extra="forbid" because of the way we use
        # kwargs to skip root_validator on some fields


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
        return obj.dict().values()
    else:
        raise TypeError(f"Unsupported type {type(obj)}")
