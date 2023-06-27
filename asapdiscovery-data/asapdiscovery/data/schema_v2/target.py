from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from asapdiscovery.data.openeye import oechem, oemol_to_pdb_string, pdb_string_to_oemol
from pydantic import Field

from .dynamic_properties import TargetType
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
        cls, pdb_file: Union[str, Path], target_name: Optional[str] = None, **kwargs
    ) -> Target:
        # directly read in data
        pdb_str = read_file_directly(pdb_file)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    @classmethod
    def from_pdb_via_openeye(
        cls, pdb_file: Union[str, Path], target_name: Optional[str] = None, **kwargs
    ) -> Target:
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
    ) -> Target:
        pdb_str = oemol_to_pdb_string(mol)
        return cls(data=pdb_str, target_name=target_name, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        return pdb_string_to_oemol(self.data)

    """
    we are deferring responsibility of writing to and from OEDesignUnit to the caller
    as it doesn't really make sense to force a specific OESpruce workflow on the user.
    as there are so many different ways to generate a OEDesignUnit from a PDB file
    depending on the options used.

    Therefore the user is responsible for reading and outputting an OEMol or OEGraphMol of the OEDesignUnit
    component in question.

    eg.

    L = Ligand.from_pdb('complex.pdb')
    prepped_oemol = prep_oemol(L.to_oemol())
    L2 = Ligand.from_oemol(prepped_oemol)
    """
