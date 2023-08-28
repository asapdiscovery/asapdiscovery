from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from asapdiscovery.data.schema_v2.complex import PreppedComplex, Complex

from typing import Any
from pydantic import Field


class PairBase(DataModelAbstractBase):
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PairBase):
            return NotImplemented

        # Just check that both Complex and Ligands are the same
        return (self.complex == other.complex) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class CompoundStructurePair(PairBase):
    complex: Complex = Field(description="Target schema object")
    ligand: Ligand = Field(description="Ligand schema object")


class DockingInputPair(PairBase):
    complex: PreppedComplex = Field(description="Target schema object")
    ligand: Ligand = Field(description="Ligand schema object")

    @classmethod
    def from_compound_structure_pair(
        cls, compound_structure_pair: CompoundStructurePair
    ) -> "DockingInputPair":
        prepped_complex = PreppedComplex.from_complex(compound_structure_pair.complex)
        return cls(complex=prepped_complex, ligand=compound_structure_pair.ligand)
