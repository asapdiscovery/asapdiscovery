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
    unassigned: list[Ligand] = Field(
        ..., description="Ligands that could not be assigned a parent"
    )

    class Config:
        allow_mutation = False

    @classmethod
    def from_ligands(
        cls, ligands: list[Ligand], no_tag: str = "ignore", infer_parents: bool = True
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

        expansions = []
        # keep track of children that have been assigned a parent
        assigned = set()
        for l1 in has_tag:
            children = {
                l2 for l2 in has_tag if l2.expansion_tag.is_child_of(l1.expansion_tag)
            }
            if len(children) > 0:
                expansion = StateExpansion(
                    parent=l1, children=children, expander={}, provenance={}
                )
                expansions.append(expansion)
                assigned.update(children)

        # check for unassigned ligands
        unassigned = [ligand for ligand in ligands if ligand not in assigned]

        if infer_parents:
            parent_ligands_unassigned = [
                Ligand.from_inchi(
                    ligand.expansion_tag.parent_inchi, compound_name="INFERRED_PARENT"
                )
                for ligand in unassigned
                if ligand.expansion_tag is not None
            ]

            # assign them a parent tag
            for l in parent_ligands_unassigned:
                _ = l.make_parent_tag()
                l.expansion_tag.inferred = True

            ligands_w_unassigned = ligands + parent_ligands_unassigned
            return cls.from_ligands(
                ligands=ligands_w_unassigned, no_tag=no_tag, infer_parents=False
            )

        return StateExpansionSet(expansions=expansions, unassigned=unassigned)

    @property
    def n_expanded_states(self) -> int:
        return sum([expansion.n_expanded_states for expansion in self.expansions])

    def to_networkx(self) -> nx.DiGraph:
        print("to_networkx")
        graphs = []
        for expansion in self.expansions:
            graph = expansion.to_networkx()
            graphs.append(graph)
        if len(graphs) == 0:
            return nx.DiGraph()
        else:
            return nx.compose_all(graphs)
