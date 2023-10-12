from typing import Any, Optional

from asapdiscovery.data.schema_v2.identifiers import LigandIdentifiers
from pydantic import BaseModel, Field


class StateExpansionTag(BaseModel):
    """
    Schema to record the expansion of a ligand state. Here we track the parent ligand and how this ligand state was
    created.

    Note we use fixed hydrogen inchikeys to distinguish between tautomers.
    """

    parent_fixed_inchikey: str = Field(
        ..., description="The fixed hydrogen map inchi key of the parent molecule."
    )
    parent_smiles: str = Field(
        ..., description="The isomeric smiles string for the parent."
    )
    parent_identifiers: Optional[LigandIdentifiers] = Field(
        ...,
        description="The set of parent identifiers which can be used to "
        "identify the parent in a workflow.",
    )
    provenance: dict[str, Any] = Field(
        ...,
        description="Provenance of the software used during the expansion and the state expander..",
    )

    def __hash__(self) -> int:
        return hash(self.json())

    # @property
    # def is_parent(self):
    #     return self.inchi == self.parent_inchi

    # @property
    # def is_child(self):
    #     return self.inchi != self.parent_inchi

    # def is_parent_of(self, other: "StateExpansionTag"):
    #     return self.inchi == other.parent_inchi
    #
    # def is_child_of(self, other: "StateExpansionTag"):
    #     return self.parent_inchi == other.inchi

    # @classmethod
    # def parent(cls, inchi: str, provenance: dict[str, str] = None):
    #     return cls(inchi=inchi, parent_inchi=inchi, provenance=provenance)

    @classmethod
    def from_parent(cls, parent, provenance: dict[str, str]):
        return cls(
            parent_fixed_inchikey=parent.fixed_inchikey,
            parent_smiles=parent.smiles,
            parent_identifiers=parent.ids,
            provenance=provenance,
        )
