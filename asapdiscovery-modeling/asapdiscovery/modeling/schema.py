from enum import Enum

from pydantic import BaseModel, Field


class MoleculeComponent(str, Enum):
    PROTEIN = "protein"
    LIGAND = "ligand"
    WATER = "water"
    OTHER = "other"


class _Model(BaseModel):
    class Config:
        extra = "forbid"


class MoleculeFilter(_Model):
    """Filter for selecting components of a molecule."""

    protein_chains: list = Field(
        list(),
        description="List of chains containing the desired protein. An empty list will return all chains.",
    )
    ligand_chain: str = Field(
        None,
        description="Chain containing the desired ligand. An empty list will return all chains.",
    )
    water_chains: list = Field(
        list(),
        description="List of chains containing the desired water. An empty list will return all chains.",
    )
    other_chains: list = Field(
        list(),
        description="List of chains containing other items. An empty list will return all chains.",
    )
    components_to_keep: list[MoleculeComponent] = Field(
        ["protein", "ligand", "water", "other"],
        description="List of components to keep. An empty list will return all components.",
    )
