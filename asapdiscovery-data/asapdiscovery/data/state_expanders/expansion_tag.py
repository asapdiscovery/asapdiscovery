from pydantic import BaseModel, Field, UUID4
import uuid


class StateExpansionTag(BaseModel):
    id: UUID4 = Field(uuid.uuid4(), description="UUID for this molecule")
    parent_id: UUID4 = Field(uuid.uuid4(), description="UUID for parent molecule")

    def __hash__(self) -> int:
        return hash(self.json())

    @property
    def is_parent(self):
        return self.id == self.parent_id

    @property
    def is_child(self):
        return self.id != self.parent_id

    def is_parent_of(self, other: "StateExpansionTag"):
        return self.id == other.parent_id

    def is_child_of(self, other: "StateExpansionTag"):
        return self.parent_id == other.id

    @classmethod
    def parent(cls):
        id = uuid.uuid4()
        return cls(id=id, parent_id=id)

    @classmethod
    def from_parent(cls, parent_tag: "StateExpansionTag"):
        return cls(parent_id=parent_tag.id)
