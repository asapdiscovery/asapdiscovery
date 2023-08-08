from typing import Literal
from openff.models.models import DefaultModel
import json
import abc


class _SchemaBase(abc.ABC, DefaultModel):
    """
    A basic schema class used to define the components of the Free energy workflow
    """

    type: Literal["base"] = "base"

    def to_file(self, filename: str):
        """
        Write the model to JSON file.
        """
        from gufe.tokenization import JSON_HANDLER
        with open(filename, "w") as output:
            json.dump(self.dict(), output, cls=JSON_HANDLER.encoder, indent=2)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the model from a JSON file
        """
        from gufe.tokenization import JSON_HANDLER
        with open(filename, "r") as f:
            return cls.parse_obj(json.load(f, cls=JSON_HANDLER.decoder))


class _SchemaBaseFrozen(_SchemaBase):
    type: Literal["_SchemaBaseFrozen"] = "_SchemaBaseFrozen"

    class Config:
        allow_mutation = False
