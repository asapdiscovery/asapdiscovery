import abc
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _SchemaBase(abc.ABC, BaseModel):
    """
    A basic schema class used to define the components of the Free energy workflow
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    type: Literal["base"] = "base"

    def to_file(self, filename: str):
        """
        Write the model to JSON file.
        """
        from gufe.tokenization import JSON_HANDLER

        with open(filename, "w") as output:
            json.dump(self.model_dump(), output, cls=JSON_HANDLER.encoder, indent=2)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the model from a JSON file
        """
        from gufe.tokenization import JSON_HANDLER

        with open(filename) as f:
            return cls.model_validate(json.load(f, cls=JSON_HANDLER.decoder))


class _SchemaBaseFrozen(_SchemaBase):
    model_config = ConfigDict(frozen=True)

    type: Literal["_SchemaBaseFrozen"] = "_SchemaBaseFrozen"
