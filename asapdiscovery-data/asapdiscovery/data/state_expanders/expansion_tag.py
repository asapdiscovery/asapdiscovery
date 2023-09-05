from typing import Any, Optional

from pydantic import BaseModel, Field


class StateExpansionTag(BaseModel):
    inchi: str = Field(..., description="UUID for this molecule")
    parent_inchi: str = Field(..., description="UUID for parent molecule")
    inferred: bool = Field(
        False,
        description="Whether this molecule has been inferred from its children or parent",
    )
    provenance: Optional[dict[str, Any]] = Field(
        ..., description="Provenance of the expansion"
    )

    def __hash__(self) -> int:
        return hash(self.json())

    @property
    def is_parent(self):
        return self.inchi == self.parent_inchi

    @property
    def is_child(self):
        return self.inchi != self.parent_inchi

    def is_parent_of(self, other: "StateExpansionTag"):
        return self.inchi == other.parent_inchi

    def is_child_of(self, other: "StateExpansionTag"):
        return self.parent_inchi == other.inchi

    @classmethod
    def parent(cls, inchi: str, provenance: dict[str, str] = None):
        return cls(inchi=inchi, parent_inchi=inchi, provenance=provenance)

    @classmethod
    def from_parent(
        cls,
        parent_tag: "StateExpansionTag",
        inchi: str,
        provenance: dict[str, str] = None,
    ):
        return cls(
            parent_inchi=parent_tag.inchi,
            inchi=inchi,
            provenance=provenance,
        )
