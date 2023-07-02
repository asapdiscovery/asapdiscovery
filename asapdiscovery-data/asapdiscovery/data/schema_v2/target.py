from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from asapdiscovery.data.openeye import oechem, oemol_to_pdb_string, pdb_string_to_oemol
from pydantic import Field, root_validator

from .dynamic_properties import TargetType
from .schema_base import (
    DataModelAbstractBase,
    DataStorageType,
    read_file_directly,
    schema_dict_get_val_overload,
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

    @root_validator(pre=True)
    @classmethod
    def _validate_at_least_one_id(cls, v):
        # check if skip validation
        if v.get("_skip_validate_ids"):
            return v
        else:
            ids = v.get("ids")
            compound_name = v.get("target_name")
            # check if all the identifiers are None, sometimes when this is called from
            # already instantiated ligand we need to be able to handle a dict and instantiated class
            if compound_name is None:
                if ids is None or all(
                    [v is None for v in schema_dict_get_val_overload(ids)]
                ):
                    raise ValueError(
                        "At least one identifier must be provide, or target_name must be provided"
                    )
        return v

    @classmethod
    def from_pdb(
        cls, pdb_file: Union[str, Path], target_name: Optional[str] = None, **kwargs
    ) -> "Target":
        # directly read in data
        pdb_str = read_file_directly(pdb_file)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    @classmethod
    def from_pdb_via_openeye(
        cls, pdb_file: Union[str, Path], target_name: Optional[str] = None, **kwargs
    ) -> "Target":
        # directly read in data
        pdb_str = read_file_directly(pdb_file)
        # NOTE: tradeof between speed and consistency with `from_pdb` method lines below will make sure that the pdb string is
        # consistent between a load and dump by roundtripping through and openeye mol but will slow down the process significantly.
        mol = pdb_string_to_oemol(pdb_str)
        pdb_str = oemol_to_pdb_string(mol)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    def to_pdb(self, filename: Union[str, Path]) -> None:
        # directly write out data
        write_file_directly(filename, self.data)

    @classmethod
    def from_oemol(
        cls, mol: oechem.OEMol, target_name: Optional[str] = None, **kwargs
    ) -> "Target":
        pdb_str = oemol_to_pdb_string(mol)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        return pdb_string_to_oemol(self.data)
