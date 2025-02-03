import logging
from collections import defaultdict
from typing import Any, ClassVar

from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class MultiStructureBase(BaseModel):
    """
    Base class for one ligand to many possible reference structures.
    """

    is_cacheable: ClassVar[bool] = False

    ligand: Ligand = Field(description="Ligand schema object")
    complexes: list[Complex] = Field(description="List of reference structures")

    @classmethod
    def _from_pairs(
        cls,
        pair_list: list[CompoundStructurePair],
    ) -> list:
        """
        Create a list of CompoundMultiStructures from a list of CompoundStructurePairs.
        Automatically separates out the ligands.
        """
        ligand_complexes_dict = defaultdict(list)

        for pair in pair_list:
            ligand = pair.ligand
            complex = pair.complex
            ligand_complexes_dict[ligand].append(complex)

        compound_multi_structures = [
            cls(ligand=ligand, complexes=complexes)
            for ligand, complexes in ligand_complexes_dict.items()
        ]

        return compound_multi_structures

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MultiStructureBase):
            raise NotImplementedError

        # Just check that both Complexs and Ligands are the same
        return (self.complexes == other.complexes) and (self.ligand == other.ligand)

    def __neq__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def complexes_hash(self):
        import hashlib

        data = ""
        for c in self.complexes:
            data += c.hash
        return hashlib.sha256(data.encode()).hexdigest()

    @property
    def unique_name(self):
        return f"{self.ligand.compound_name}-{self.ligand.fixed_inchikey}_{self.complexes_hash}"


class CompoundMultiStructure(MultiStructureBase):
    """
    Schema for one ligand to many possible reference structures.
    """

    @classmethod
    def from_pairs(
        cls, pair_list: list[CompoundStructurePair]
    ) -> list["CompoundMultiStructure"]:
        return cls._from_pairs(pair_list)
