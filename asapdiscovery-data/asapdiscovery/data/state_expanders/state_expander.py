import abc
from typing import Any, Literal

import networkx as nx
from asapdiscovery.data.schema_v2.ligand import Ligand
from pydantic import BaseModel, Field


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

    # could split into a class with List[StateExpansion] but seems like overkill

    @staticmethod
    def to_networkx(expansions: list["StateExpansion"]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for expansion in expansions:
            graph.add_node(expansion.parent)
            for child in expansion.children:
                graph.add_node(child)
                graph.add_edge(expansion.parent, child)
        return graph

    @staticmethod
    def ligands_to_networkx(ligands: list[Ligand]) -> nx.DiGraph:
        parents = [ligand for ligand in ligands if ligand.expansion_tag.is_parent]
        expansions = []
        for parent in parents:
            children = [
                ligand
                for ligand in ligands
                if ligand.expansion_tag.is_child_of(parent.expansion_tag)
                and not ligand.expansion_tag.is_parent
            ]
            expansion = StateExpansion(
                parent=parent, children=children, expander={}, provenance={}
            )
            expansions.append(expansion)

        return StateExpansion.to_networkx(expansions=expansions)
