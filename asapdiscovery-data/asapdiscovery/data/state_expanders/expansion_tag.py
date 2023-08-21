from typing import Any, Optional

from pydantic import BaseModel, Field


class StateExpansionTag(BaseModel):
    inchikey: str = Field(..., description="UUID for this molecule")
    parent_inchikey: str = Field(..., description="UUID for parent molecule")
    provenance: Optional[dict[str, Any]] = Field(
        ..., description="Provenance of the expansion"
    )

    def __hash__(self) -> int:
        return hash(self.json())

    @property
    def is_parent(self):
        return self.inchikey == self.parent_inchikey

    @property
    def is_child(self):
        return self.inchikey != self.parent_inchikey

    def is_parent_of(self, other: "StateExpansionTag"):
        return self.inchikey == other.parent_inchikey

    def is_child_of(self, other: "StateExpansionTag"):
        return self.parent_inchikey == other.inchikey

    @classmethod
    def parent(cls, inchikey: str, provenance: dict[str, str] = None):
        return cls(inchikey=inchikey, parent_inchikey=inchikey, provenance=provenance)

    @classmethod
    def from_parent(
        cls,
        parent_tag: "StateExpansionTag",
        inchikey: str,
        provenance: dict[str, str] = None,
    ):
        return cls(
            parent_inchikey=parent_tag.inchikey,
            inchikey=inchikey,
            provenance=provenance,
        )
