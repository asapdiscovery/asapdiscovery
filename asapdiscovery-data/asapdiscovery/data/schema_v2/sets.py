from typing import Any

from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from pydantic import Field


class MultiStructureBase(DataModelAbstractBase):
    """
    Base class for one ligand to many possible reference structures.
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MultiStructureBase):
            return NotImplemented

        # Just check that the Ligands and sets of Complexes are the same
        return (self.ligand == other.ligand) and (
            set(self.complex) == set(other.complex)
        )


class LigandMultiStructure(MultiStructureBase):
    """
    Schema for one ligand to many possible reference structures.
    """

    ligand: Ligand = Field(description="Ligand schema object")
    complexes: list[Complex] = Field(description="List of reference structures")


class DockingInputMultiStructure(MultiStructureBase):
    """
    Schema for one ligand to many possible reference structures.
    """

    ligand: Ligand = Field(description="Ligand schema object")
    complexes: list[PreppedComplex] = Field(description="List of reference structures")
