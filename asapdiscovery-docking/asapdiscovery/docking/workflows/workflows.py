"""
Schema for workflows base classes
"""

import logging
from pathlib import Path
from typing import Optional

from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from pydantic import BaseModel, Field, PositiveInt, root_validator


class DockingWorkflowInputsBase(BaseModel):
    ligands: Optional[str] = Field(
        None, description="Path to a molecule file containing query ligands."
    )

    pdb_file: Optional[Path] = Field(
        None, description="Path to a PDB file to prep and dock to."
    )

    fragalysis_dir: Optional[Path] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )
    structure_dir: Optional[Path] = Field(
        None,
        description="Path to a directory containing structures to dock instead of a full fragalysis database.",
    )

    cache_dir: Optional[str] = Field(
        None, description="Path to a directory where a cache has been generated"
    )

    use_only_cache: bool = Field(
        False,
        description="Whether to only use the cached structures, otherwise try to prep uncached structures.",
    )

    save_to_cache: bool = Field(
        True,
        description="Generate a cache from structures prepped in this workflow run in this directory",
    )

    target: TargetTags = Field(None, description="The target to dock against.")

    write_final_sdf: bool = Field(
        default=True,
        description="Whether to write the final docked poses to an SDF file.",
    )
    use_dask: bool = Field(True, description="Whether to use dask for parallelism.")

    dask_type: DaskType = Field(
        DaskType.LOCAL, description="Dask client to use for parallelism."
    )

    dask_cluster_n_workers: PositiveInt = Field(
        10,
        description="Number of workers to use as inital guess for Lilac dask cluster",
    )

    dask_cluster_max_workers: PositiveInt = Field(
        200, description="Maximum number of workers to use for Lilac dask cluster"
    )

    n_select: PositiveInt = Field(
        5, description="Number of targets to dock each ligand against."
    )
    logname: str = Field(
        "", description="Name of the log file."
    )  # use root logger for proper forwarding of logs from dask

    loglevel: int = Field(logging.DEBUG, description="Logging level")

    output_dir: Path = Field(Path("output"), description="Output directory")

    overwrite: bool = Field(
        False, description="Whether to overwrite existing output directory."
    )
    walltime: str = Field(
        "72h", description="Walltime for the workflow, used for dask-jobqueue"
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_json_file(cls, file: str | Path):
        return cls.parse_file(str(file))

    def to_json_file(self, file: str | Path):
        with open(file, "w") as f:
            f.write(self.json(indent=2))

    @root_validator
    @classmethod
    def check_inputs(cls, values):
        """
        Validate inputs
        """
        ligands = values.get("ligands")
        fragalysis_dir = values.get("fragalysis_dir")
        structure_dir = values.get("structure_dir")
        postera = values.get("postera")
        pdb_file = values.get("pdb_file")

        if postera and ligands:
            raise ValueError("Cannot specify both ligands and postera.")

        if not postera and not ligands:
            raise ValueError("Must specify either ligands or postera.")

        # can only specify one of fragalysis dir, structure dir and PDB file
        if sum([bool(fragalysis_dir), bool(structure_dir), bool(pdb_file)]) != 1:
            raise ValueError(
                "Must specify exactly one of fragalysis_dir, structure_dir or pdb_file"
            )

        return values


class PosteraDockingWorkflowInputs(DockingWorkflowInputsBase):
    postera: bool = Field(
        False, description="Whether to use the Postera database as the query set."
    )

    postera_upload: bool = Field(
        False, description="Whether to upload the results to Postera."
    )
    postera_molset_name: Optional[str] = Field(
        None, description="The name of the molecule set to upload to."
    )
