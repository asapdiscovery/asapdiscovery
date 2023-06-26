from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401
from enum import Enum
from asapdiscovery.data.openeye import (
    oechem,
    oemol_to_pdb_string,
    pdb_string_to_oemol,
    oedu_to_pdb_string,
    pdb_string_to_oedu,
)
from .dynamic_properties import TargetType
from pydantic import Field

from .schema_base import (
    DataModelAbstractBase,
    DataStorageType,
    read_file_directly,
    write_file_directly,
)


class InvalidTargetError(ValueError):
    ...


class TargetIdentifiers(DataModelAbstractBase):
    """
    Identifiers for a Ligand
    """

    target_type: Optional[TargetType] = Field(
        None,
        description="Dynamic Enum describing the target type e.g sars2, mers or mac1",
    )

    fragalysis_id: Optional[str] = Field(
        None, description="The PDB code of the target if applicable"
    )

    pdb_code: Optional[str] = Field(
        None, description="The PDB code of the target if applicable"
    )


class Target(DataModelAbstractBase):
    """
    Schema for a Target
    """

    target_name: str = Field(None, description="The name of the target")

    ids: Optional[TargetIdentifiers] = Field(
        None,
        description="TargetIdentifiers Schema for identifiers associated with this ligand",
    )

    data: str = Field(
        "",
        description="PDB file stored as a string to hold internal data state",
        repr=False,
    )
    data_format: DataStorageType = Field(
        DataStorageType.pdb,
        description="Enum describing the data storage method",
        allow_mutation=False,
    )

    @classmethod
    def from_pdb(
        cls, pdb_file: Union[str, Path], target_name: str | None = None, **kwargs
    ) -> Target:
        # directly read in data
        pdb_str = read_file_directly(pdb_file)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    def to_pdb(self, filename: Union[str, Path]) -> None:
        # directly write out data
        write_file_directly(filename, self.data)

    @classmethod
    def from_oemol(
        cls, mol: oechem.OEMol, target_name: str | None = None, **kwargs
    ) -> Target:
        pdb_str = oemol_to_pdb_string(mol)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        return pdb_string_to_oemol(self.data)

    @classmethod
    def from_oedu(
        cls, du_file: Union[str, Path], target_name: str | None = None, **kwargs
    ) -> Target:
        pdb_str = oedu_to_pdb_string(du_file)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    def to_oedu(self) -> oechem.OEDesignUnit:
        return pdb_string_to_oedu(self.data)
