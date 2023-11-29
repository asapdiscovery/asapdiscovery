import logging
from collections import defaultdict

from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from pydantic import Field

logger = logging.getLogger(__name__)


class MultiStructureBase(DataModelAbstractBase):
    """
    Base class for one ligand to many possible reference structures.
    """

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


class CompoundMultiStructure(MultiStructureBase):
    """
    Schema for one ligand to many possible reference structures.
    """

    ligand: Ligand = Field(description="Ligand schema object")
    complexes: list[Complex] = Field(description="List of reference structures")

    @classmethod
    def from_pairs(
        cls, pair_list: list[CompoundStructurePair]
    ) -> list["CompoundMultiStructure"]:
        return cls._from_pairs(pair_list)
