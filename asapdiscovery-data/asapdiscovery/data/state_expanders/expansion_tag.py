from pydantic import UUID4, BaseModel, Field

class StateExpansionTag(BaseModel):
    inchikey: str = Field(None, description="UUID for this molecule")
    parent_inchikey: str = Field(None, description="UUID for parent molecule")

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
    def parent(cls, inchikey: str):
        return cls(inchikey=inchikey, parent_inchikey=inchikey)

    @classmethod
    def from_parent(cls, parent_tag: "StateExpansionTag", inchikey: str):
        return cls(parent_inchikey=parent_tag.inchikey, inchikey=inchikey)
