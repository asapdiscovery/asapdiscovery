import abc
from typing import Any, Literal, List

import networkx as nx
from asapdiscovery.data.schema_v2.ligand import Ligand
from pydantic import BaseModel, Field


class StateExpanderBase(abc.ABC, BaseModel):
    expander_type: Literal["StateExpanderBase"] = Field(
        "StateExpanderBase", description="The type of expander to use"
    )

    @abc.abstractmethod
    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        ...

    def expand(self, ligands: list[Ligand]) -> list[Ligand]:
        return self._expand(ligands=ligands)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


class StateExpansion(BaseModel):
    parent: Ligand = Field(..., description="The parent ligand")
    children: list[Ligand] = Field(
        ..., description="The children ligands resulting from expansion"
    )

    class Config:
        allow_mutation = False

    @property
    def n_expanded_states(self) -> int:
        return len(self.children)

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for expansion in expansions:
            graph.add_node(expansion.parent)
            for child in expansion.children:
                graph.add_node(child)
                graph.add_edge(expansion.parent, child)
        return graph


class StateExpansionSet(BaseModel):
    expansions: List[StateExpansion] = Field(..., description="The set of expansions")

    class Config:
        allow_mutation = False

    @classmethod
    def from_ligands(
        ligands: List[Ligand], no_tag: str = "ignore"
    ) -> "StateExpansionSet":
        has_tag = [ligand.expansion_tag is not None for ligand in ligands]
        if not all(has_tag):
            if no_tag == "ignore":
                pass
            elif no_tag == "raise":
                raise ValueError("Some ligands do not have an expansion tag")
            else:
                raise ValueError(
                    f"Unknown value for no_tag: {no_tag}, must be 'ignore' or 'raise'"
                )

        parents = [ligand for ligand in has_tag if ligand.expansion_tag.is_parent]
        expansions = []
        for parent in parents:
            children = [
                ligand
                for ligand in has_tag
                if ligand.expansion_tag.is_child_of(parent.expansion_tag)
            ]
            expansion = StateExpansion(
                parent=parent, children=children, expander={}, provenance={}
            )
            expansions.append(expansion)

        return StateExpansionSet(expansions=expansions)

    @property
    def n_expanded_states(self) -> int:
        return sum([expansion.n_expanded_states for expansion in self.expansions])

    def to_networkx(self) -> nx.DiGraph:
        graphs = []
        for expansion in self.expansions:
            graph = expansion.to_networkx()
            graphs.append(graph)

        return nx.compose_all(graphs)
