"""
Schema for workflows base classes
"""
import abc
import logging
from pathlib import Path
from typing import Optional

from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.modeling.protein_prep_v2 import CacheType
from pydantic import BaseModel, Field, PositiveInt, root_validator, validator


class WorkflowInputsBase(BaseModel):
    filename: Optional[str] = Field(
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
    postera: bool = Field(
        False, description="Whether to use the Postera database as the query set."
    )

    cache_dir: Optional[str] = Field(
        None, description="Path to a directory where a cache has been generated"
    )

    gen_cache: Optional[str] = Field(
        None,
        description="Generate a cache from structures prepped in this workflow run in this directory",
    )

    cache_type: Optional[list[str]] = Field(
        [CacheType.DesignUnit], description="The types of cache to use."
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

    # TODO: make this an "abstractmethod" kinda thing that needs to be re-named in subclasses
    logname: str = Field("docking_workflow_base", description="Name of the log file.")

    loglevel: int = Field(logging.INFO, description="Logging level")

    output_dir: Path = Field(Path("output"), description="Output directory")

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
        filename = values.get("filename")
        fragalysis_dir = values.get("fragalysis_dir")
        structure_dir = values.get("structure_dir")
        postera = values.get("postera")
        cache_dir = values.get("cache_dir")
        gen_cache = values.get("gen_cache")
        pdb_file = values.get("pdb_file")

        if postera and filename:
            raise ValueError("Cannot specify both filename and postera.")

        if not postera and not filename:
            raise ValueError("Must specify either filename or postera.")

        # can only specify one of fragalysis dir, structure dir and PDB file
        if sum([bool(fragalysis_dir), bool(structure_dir), bool(pdb_file)]) != 1:
            raise ValueError(
                "Must specify exactly one of fragalysis_dir, structure_dir or pdb_file"
            )

        if cache_dir and gen_cache:
            raise ValueError("Cannot specify both cache_dir and gen_cache.")
        return values

    @validator("cache_dir")
    @classmethod
    def cache_dir_must_be_directory(cls, v):
        """
        Validate that the DU cache is a directory
        """
        if v is not None:
            if not Path(v).is_dir():
                raise ValueError("Du cache must be a directory.")
        return v