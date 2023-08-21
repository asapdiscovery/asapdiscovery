import abc
from typing import Literal

import networkx as nx
from asapdiscovery.data.schema_v2.ligand import Ligand
from pydantic import BaseModel, Field


class StateExpanderBase(abc.ABC, BaseModel):
    expander_type: Literal["StateExpanderBase"] = Field(
        "StateExpanderBase", description="The type of expander to use"
    )

    @abc.abstractmethod
    def _expand(self, ligands: list[Ligand], unique: bool = False) -> list[Ligand]:
        ...

    def expand(self, ligands: list[Ligand], unique: bool = True) -> list[Ligand]:
        expanded_ligands = self._expand(ligands=ligands)
        if unique:
            return list(set(expanded_ligands))
        else:
            return expanded_ligands

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
        graph.add_node(self.parent)
        for child in self.children:
            graph.add_node(child)
            graph.add_edge(self.parent, child)
        return graph


class StateExpansionSet(BaseModel):
    expansions: list[StateExpansion] = Field(..., description="The set of expansions")

    class Config:
        allow_mutation = False

    @classmethod
    def from_ligands(
        cls, ligands: list[Ligand], no_tag: str = "ignore"
    ) -> "StateExpansionSet":
        has_tag = [ligand for ligand in ligands if ligand.expansion_tag is not None]
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
            children = {
                ligand
                for ligand in has_tag
                if ligand.expansion_tag.is_child_of(parent.expansion_tag)
            }

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
