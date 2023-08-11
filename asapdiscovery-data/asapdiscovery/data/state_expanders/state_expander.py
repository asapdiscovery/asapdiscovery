import abc
from typing import Any, Literal

from pydantic import BaseModel, Field

from asapdiscovery.data.schema_v2.ligand import Ligand


class StateExpanderBase(abc.ABC, BaseModel):
    expander_type: Literal["StateExpanderBase"] = Field(
        "StateExpanderBase", description="The type of expander to use"
    )

    @abc.abstractmethod
    def _expand(self, ligands: list[Ligand]) -> list["StateExpansion"]:
        ...

    def expand(self, ligands: list[Ligand]) -> list["StateExpansion"]:
        return self._expand(ligands=ligands)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


class StateExpansion(BaseModel):
    parent: Ligand = Field(..., description="The parent ligand")
    children: list[Ligand] = Field(
        ..., description="The children ligands resulting from expansion"
    )
    expander: dict[str, Any] = Field(
        ..., description="The expander and settings used to enumerate the states."
    )
    provenance: dict[str, str] = Field(
        ...,
        description="The provenance of the state expander which worked on this molecule.",
    )

    class Config:
        allow_mutation = False

    @property
    def n_expanded_states(self) -> int:
        return len(self.children)

    @property
    def parent_smiles(self) -> str:
        return self.parent.smiles

    @staticmethod
    def flatten_children(expansions: list["StateExpansion"]) -> list[Ligand]:
        return [child for expansion in expansions for child in expansion.children]

    @staticmethod
    def flatten_parents(expansions: list["StateExpansion"]) -> list[Ligand]:
        return [expansion.parent for expansion in expansions]
