from typing import Any

from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from pydantic import Field


class PairBase(DataModelAbstractBase):
    """
    Base class for pairs.
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PairBase):
            return NotImplemented

        # Just check that both Complex and Ligands are the same
        return (self.complex == other.complex) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class CompoundStructurePair(PairBase):
    """
    Schema for a CompoundStructurePair, containing both a Complex and Ligand
    This is designed to track a matched ligand and complex pair for investigation
    """

    complex: Complex = Field(description="Target schema object")
    ligand: Ligand = Field(description="Ligand schema object")
