import abc
from typing import Literal

from asapdiscovery.data.schema.ligand import Ligand
from pydantic.v1 import BaseModel, Field


class StateExpanderBase(abc.ABC, BaseModel):
    expander_type: Literal["StateExpanderBase"] = Field(
        "StateExpanderBase", description="The type of expander."
    )

    @abc.abstractmethod
    def _expand(self, ligands: list[Ligand], unique: bool = False) -> list[Ligand]: ...

    def expand(self, ligands: list[Ligand], unique: bool = True) -> list[Ligand]:
        expanded_ligands = self._expand(ligands=ligands)
        if unique:
            return list(set(expanded_ligands))
        else:
            return expanded_ligands

    @abc.abstractmethod
    def _provenance(self) -> dict[str, str]:
        """Return the software used to perform the state expansion in the workflow."""
        ...

    def provenance(self) -> dict[str, str]:
        """
        Get the provenance of the software and settings used to expand the molecule state.
        Returns
        -------
            A dict of the expander and the software used to do the expansion.
        """
        data = {"expander": self.dict()}
        data.update(self._provenance())
        return data


class StateExpansion(BaseModel):
    parent: Ligand = Field(..., description="The parent ligand")
    children: list[Ligand] = Field(
        ..., description="The children ligands resulting from expansion"
    )
    expansion: Literal["stereo", "charge"] = Field(
        ...,
        description="The type of state expansion, this will be used "
        "to group the expansions.",
    )

    class Config:
        allow_mutation = False

    @property
    def n_expanded_states(self) -> int:
        return len(self.children)


class StateExpansionSet(BaseModel):
    expansions: list[StateExpansion] = Field(..., description="The set of expansions")
    unassigned: list[Ligand] = Field(
        ..., description="Ligands that could not be assigned a parent"
    )

    class Config:
        allow_mutation = False

    @classmethod
    def from_ligands(cls, ligands: list[Ligand]) -> "StateExpansionSet":
        is_expansion = [
            ligand for ligand in ligands if ligand.expansion_tag is not None
        ]

        expansions = []
        # keep track of children that have been assigned a parent
        assigned = set()
        for ligand in ligands:
            inchikey = ligand.fixed_inchikey
            children = [
                child
                for child in is_expansion
                if child.expansion_tag.parent_fixed_inchikey == inchikey
            ]
            if len(children) > 0:
                # work out the type of expansion, make sure only one type links the children and parents
                expansion_type = [
                    child.expansion_tag.provenance["expander"]["expander_type"].lower()
                    for child in children
                ]
                if len(set(expansion_type)) > 1:
                    raise RuntimeError(
                        f"Multiple expansion methods link the parent {ligand.smiles} to the child molecules {[child.smiles for child in children]} this should not happen."
                    )
                # set the type to one of the two defined types
                expansion_method = (
                    "stereo"
                    if expansion_type[0].lower().find("stereo") == 0
                    else "charge"
                )

                expansion = StateExpansion(
                    parent=ligand, children=children, expansion=expansion_method
                )
                expansions.append(expansion)
                assigned.update(children)
                assigned.add(ligand)

        # check for unassigned ligands
        unassigned = [ligand for ligand in ligands if ligand not in assigned]

        return StateExpansionSet(expansions=expansions, unassigned=unassigned)

    def get_stereo_expansions(self) -> list[StateExpansion]:
        return [
            expansion
            for expansion in self.expansions
            if expansion.expansion == "stereo"
        ]

    def get_charge_expansions(self) -> list[StateExpansion]:
        return [
            expansion
            for expansion in self.expansions
            if expansion.expansion == "charge"
        ]
