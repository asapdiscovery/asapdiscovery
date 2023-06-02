from enum import Enum
from pathlib import Path

from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema import CrystalCompoundData, CrystalCompoundDataset
from pydantic import BaseModel, Field


class MoleculeComponent(str, Enum):
    PROTEIN = "protein"
    LIGAND = "ligand"
    WATER = "water"
    OTHER = "other"


class MoleculeFilter(BaseModel):
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


class PreppedTarget(BaseModel):
    source: CrystalCompoundData = Field(None, description="Structure that was prepped")
    output_name: str = Field(None, description="Name to give to output files.")
    prepped: bool = Field(False, description="Has the target been prepped yet?")
    saved: bool = Field(False, description="Have the results been saved?")
    molecule_filter: MoleculeFilter
    output_dir: Path = Field(description="Output path for serialization")
    sdf: Path = Field(None, description="Path to prepped sdf file")
    complex: Path = Field(None, description="Path to prepped complex")
    protein: Path = Field(None, description="Path to prepped protein-only file")
    design_unit: Path = Field(None, description="Path to design unit")

    def set_prepped(self):
        self.prepped = True

    def set_saved(self):
        self.saved = True

    def get_output_files(self):
        if "ligand" in self.molecule_filter.components_to_keep:
            self.sdf = self.output_dir / f"{self.output_name}.sdf"
            self.complex = self.output_dir / f"{self.output_name}-complex.pdb"
        self.protein = self.output_dir / f"{self.output_name}-protein.pdb"
        self.design_unit = self.output_dir / f"{self.output_name}.oedu"
