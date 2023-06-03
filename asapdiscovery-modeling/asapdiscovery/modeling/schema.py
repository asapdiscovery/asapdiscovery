from enum import Enum
from pathlib import Path

from asapdiscovery.data.schema import CrystalCompoundData, Dataset
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


class PrepOpts(BaseModel):
    ref_fn: Path = Field(None, description="Reference structure to align to.")
    ref_chain: str = Field(None, description="Chain ID to align to.")
    seqres_yaml: Path = Field(None, description="Path to seqres yaml")
    loop_db: Path = Field(None, description="Path to loop database to use for prepping")
    output_dir: Path = Field(None, description="Directory to save output to.")


class PreppedTarget(BaseModel):
    source: CrystalCompoundData = Field(description="Source of model")
    output_name: str = Field(None, description="Name of output structure.")
    active_site_chain: str = Field(
        None, description="Chain identifying the active site of interest."
    )
    active_site: str = Field(None, description="OpenEye formatted active site residue.")
    lig_chain: str = Field(None, description="Chain identifying the ligand.")
    prepped: bool = Field(False, description="Has the target been prepped yet?")
    saved: bool = Field(False, description="Have the results been saved?")
    molecule_filter: MoleculeFilter
    ligand: Path = Field(None, description="Path to prepped sdf file")
    complex: Path = Field(None, description="Path to prepped complex")
    protein: Path = Field(None, description="Path to prepped protein-only file")
    design_unit: Path = Field(None, description="Path to design unit")

    def set_prepped(self):
        self.prepped = True

    def set_saved(self):
        self.saved = True

    def get_output_files(self, output_dir: Path):
        if "ligand" in self.molecule_filter.components_to_keep:
            self.ligand = output_dir / f"{self.output_name}-prepped_ligand.sdf"
            self.complex = output_dir / f"{self.output_name}-prepped_complex.pdb"
        self.protein = output_dir / f"{self.output_name}-prepped_protein.pdb"
        self.design_unit = output_dir / f"{self.output_name}-prepped_receptor.oedu"


class PreppedTargets(Dataset):
    data_type = PreppedTarget
    iterable: list[data_type]
