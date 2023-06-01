from pydantic import BaseModel, Field


class MoleculeFilter(BaseModel):
    protein_chains: list = Field(
        list(),
        description="List of chains containing the desired protein. An empty list will return all chains.",
    )
    ligand_chains: list = Field(
        list(),
        description="List of chains containing the desired ligand. An empty list will return all chains.",
    )
    water_chains: list = Field(
        list(),
        description="List of chains containing the desired water. An empty list will return all chains.",
    )
    other_chains: list = Field(
        list(),
        description="List of chains containing other items. An empty list will return all chains.",
    )
    components_to_keep: list = Field(
        ["protein", "ligand", "water", "other"],
        description="List of components to keep. An empty list will return all components.",
    )
