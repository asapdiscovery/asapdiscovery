"""
Defines docking base schema.
"""

import logging
from asapdiscovery.data.dask_utils import (
    DaskType,
)

from asapdiscovery.data.postera.manifold_data_validation import (
    TargetTags,
)
from asapdiscovery.modeling.protein_prep_v2 import CacheType

from pydantic import validator

import abc

from pathlib import Path
from typing import Literal, Optional, Union

import dask
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    oechem,
)
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import DockingInputPair

from asapdiscovery.modeling.modeling import split_openeye_design_unit
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, root_validator


class DockingInputsBase(BaseModel):
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

    logname: str = Field("cross_docking", description="Name of the log file.")

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

    @abc.abstractmethod()
    @classmethod
    def _check_inputs(cls, values):
        """
        Subclass-specific input validation.
        """
        return values

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
        cls._check_inputs(values)
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


class DockingResult(BaseModel):
    """
    Schema for a DockingResult, containing both a DockingInputPair used as input to the workflow
    and a Ligand object containing the docked pose.
    Also contains the probability of the docked pose if applicable.

    Parameters
    ----------
    input_pair : DockingInputPair
        Input pair
    posed_ligand : Ligand
        Posed ligand
    probability : float, optional
        Probability of the docked pose, by default None
    provenance : dict[str, str]
        Provenance information

    """

    type: Literal["DockingResult"] = "DockingResult"
    input_pair: DockingInputPair = Field(description="Input pair")
    posed_ligand: Ligand = Field(description="Posed ligand")
    probability: Optional[PositiveFloat] = Field(
        description="Probability"
    )  # not easy to get the probability from rescoring
    provenance: dict[str, str] = Field(description="Provenance")

    def get_output(self) -> dict:
        """
        return a dictionary of some of the fields of the DockingResult
        """
        dct = self.dict()
        dct.pop("input_pair")
        dct.pop("posed_ligand")
        dct.pop("type")
        return dct

    def to_posed_oemol(self) -> oechem.OEMol:
        """
        Combine the original target and posed ligand into a single oemol

        Returns
        -------
        oechem.OEMol
            Combined oemol
        """
        _, prot, _ = split_openeye_design_unit(self.input_pair.complex.target.to_oedu())
        return combine_protein_ligand(prot, self.posed_ligand.to_oemol())

    @staticmethod
    def make_df_from_docking_results(results: list["DockingResult"]):
        """
        Make a dataframe from a list of DockingResults

        Parameters
        ----------
        results : list[DockingResult]
            List of DockingResults

        Returns
        -------
        pd.DataFrame
            Dataframe of DockingResults
        """
        import pandas as pd

        return pd.DataFrame([r.get_output() for r in results])


class DockingBase(BaseModel):
    """
    Base class for running docking
    """

    type: Literal["DockingBase"] = "DockingBase"

    @abc.abstractmethod
    def _dock() -> list[DockingResult]:
        ...

    def dock(
        self, inputs: list[DockingInputPair], use_dask: bool = False, dask_client=None
    ) -> Union[list[dask.delayed], list[DockingResult]]:
        if use_dask:
            delayed_outputs = []
            for inp in inputs:
                out = dask.delayed(self._dock)(inputs=[inp])
                delayed_outputs.append(out[0])  # flatten
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client=dask_client, errors="skip"
            )
        else:
            outputs = self._dock(inputs=inputs)
        # filter out None values
        outputs = [o for o in outputs if o is not None]
        return outputs

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...
