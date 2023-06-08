from enum import Enum
from pathlib import Path

from asapdiscovery.data.schema import CrystalCompoundData, Dataset
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


class PrepOpts(_Model):
    ref_fn: Path = Field(None, description="Reference structure to align to.")
    ref_chain: str = Field(None, description="Chain ID to align to.")
    seqres_yaml: Path = Field(None, description="Path to seqres yaml")
    loop_db: Path = Field(None, description="Path to loop database to use for prepping")
    make_design_unit: bool = Field(
        True,
        description="Whether to make a design unit or just to save the spruced protein.",
    )
    output_dir: Path = Field(None, description="Directory to save output to.")


class PreppedTarget(_Model):
    source: CrystalCompoundData = Field(description="Source of model")

    # Filtering and Prepping options
    molecule_filter: MoleculeFilter
    active_site_chain: str = Field(
        None, description="Chain identifying the active site of interest."
    )
    oe_active_site_residue: str = Field(
        None, description="OpenEye formatted active site residue."
    )
    lig_chain: str = Field(None, description="Chain identifying the ligand.")

    # Success Tracking
    prepped: bool = Field(False, description="Has the target been prepped yet?")
    saved: bool = Field(False, description="Have the results been saved?")
    failed: bool = Field(None, description="Did the prep fail?")

    # Output Files
    output_dir: Path = Field(None, description="Path to output directory.")
    output_name: str = Field(None, description="Name of output structure.")
    ligand: Path = Field(None, description="Path to prepped sdf file")
    complex: Path = Field(None, description="Path to prepped complex")
    protein: Path = Field(None, description="Path to prepped protein-only file")
    design_unit: Path = Field(None, description="Path to design unit")

    def set_prepped(self):
        self.prepped = True

    def set_saved(self):
        self.saved = True

    def get_output_files(self, success: bool, output_dir: Path = None):
        """
        Get output files for a prepped target.

        Parameters
        ----------
        success : bool
            Whether or not the prep was successful.
        output_dir : Path
            Directory to save output files to.

        Returns
        -------
        None

        """
        if not output_dir and not self.output_dir:
            raise ValueError("Must provide output directory.")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if success:
            self.failed = False
            if "ligand" in self.molecule_filter.components_to_keep:
                self.ligand = self.output_dir / f"{self.output_name}-prepped_ligand.sdf"
                self.complex = (
                    self.output_dir / f"{self.output_name}-prepped_complex.pdb"
                )
            self.protein = self.output_dir / f"{self.output_name}-prepped_protein.pdb"
            self.design_unit = (
                self.output_dir / f"{self.output_name}-prepped_receptor.oedu"
            )
        else:
            self.failed = True
            self.protein = self.output_dir / f"{self.output_name}-failed-spruced.pdb"


class PreppedTargets(Dataset):
    data_type = PreppedTarget
    iterable: list[data_type]
