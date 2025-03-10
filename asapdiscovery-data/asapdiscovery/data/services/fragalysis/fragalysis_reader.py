from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List  # noqa: F401

import dask
import pandas
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.util.dask_utils import (
    FailureMode,
    actualise_dask_delayed_iterable,
)
from pydantic.v1 import BaseModel, Field, root_validator, validator

logger = logging.getLogger(__name__)


class FragalysisFactory(BaseModel):
    """
    Schema for a loading a Fragalysis dump. The directory given by parent_dir should
    contain (at a minimum):
    1. a metadata.csv file with the columns "crystal_name" and "alternate_name"
    2. an aligned/ subdirectory, containing a subdirectory for each entry in
    metadata.csv, with these subdirectories containing the bound PDB files
    """

    parent_dir: Path = Field(
        description="Top level directory of the Fragalysis database."
    )
    xtal_col: str = Field("crystal_name", description="Name of the crystal column.")
    compound_col: str = Field(
        "alternate_name", description="Name of the compound column."
    )
    fail_missing: bool = Field(False, description="Whether to fail on missing files.")
    metadata_csv_name: str = Field(
        "metadata.csv", description="Name of the metadata file."
    )

    @validator("parent_dir")
    @classmethod
    def _validate_parent_dir(cls, v):
        if not v.exists():
            raise ValueError("Given parent_dir does not exist.")

        if not v.is_dir():
            raise ValueError("Given parent_dir is not a directory.")

        return v

    @root_validator
    @classmethod
    def _validate_metadata_csv_name(cls, values):
        parent_dir = values.get("parent_dir")
        metadata_csv_name = values.get("metadata_csv_name")
        csv_path = parent_dir / metadata_csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"No {csv_path.name} file found in parent_dir.")
        return values

    @root_validator
    @classmethod
    def _validate_aligned_dir(cls, values):
        parent_dir = values.get("parent_dir")
        aligned_dir = parent_dir / "aligned"
        if not aligned_dir.exists():
            raise FileNotFoundError("No aligned/ directory found in parent_dir.")
        return values

    def load(
        self, use_dask=False, dask_client=None, failure_mode=FailureMode.SKIP
    ) -> list[Complex]:
        """
        Load a Fragalysis dump as a list of Complex objects.

        Parameters
        ----------
        use_dask : bool, optional
            Whether to use dask to parallelise loading of PDB files.
            Defaults to False.
        dask_client : dask.distributed.Client, optional
            Dask client to use for parallelisation. Defaults to None.
        failure_mode : FailureMode
            The failure mode for dask. Can be 'raise' or 'skip'.

        Returns
        -------
        List[Complex]
            List of Complex objects.
        """
        df = pandas.read_csv(self.parent_dir / self.metadata_csv_name)

        if len(df) == 0:
            raise ValueError(f"{self.metadata_csv_name} file is empty.")

        if (self.xtal_col not in df.columns) or (self.compound_col not in df.columns):
            raise ValueError(
                f"{self.metadata_csv_name} file must contain a crystal name column and a "
                "compound name column."
            )

        all_xtal_dirs = os.listdir(self.parent_dir / "aligned")

        # Subset metadata to only contain rows with directories in aligned/
        df = df.loc[df[self.xtal_col].isin(all_xtal_dirs), :]
        if df.shape[0] == 0:
            raise ValueError(
                f"No aligned directories found with entries in {self.metadata_csv_name}."
            )

        # assign delay processing function if using dask
        if use_dask:
            parse_fn = dask.delayed(self.process_fragalysis_pdb)

        else:
            parse_fn = self.process_fragalysis_pdb

        # Loop through directories and load each bound file
        complexes = []
        for _, (xtal_name, compound_name) in df[
            [self.xtal_col, self.compound_col]
        ].iterrows():
            c = parse_fn(
                self.parent_dir,
                xtal_name,
                compound_name,
                fail_missing=self.fail_missing,
            )

            complexes.append(c)

        if use_dask:
            complexes = actualise_dask_delayed_iterable(
                complexes, dask_client=dask_client, errors=failure_mode
            )

        # remove None values
        complexes = [c for c in complexes if c is not None]

        return complexes

    @staticmethod
    def process_fragalysis_pdb(
        parent_dir, xtal_name, compound_name, fail_missing=False
    ):
        """
        Process a PDB file from a Fragalysis dump.

        Parameters
        ----------
        parent_dir : Path
            Top-level directory of the Fragalysis database
        xtal_name : str
            Name of the crystal
        compound_name : str
            Name of the compound
        fail_missing : bool, default=False
            If True, raises an error if a PDB file isn't found where expected, or a
            found PDB file can't be parsed

        Returns
        -------
        Complex
        """
        pdb_fn = parent_dir / "aligned" / xtal_name / f"{xtal_name}_bound.pdb"
        if not pdb_fn.exists():
            if fail_missing:
                raise FileNotFoundError(f"No PDB file found for {xtal_name}.")
            else:
                logger.warn(f"No PDB file found for {xtal_name}.")
                return None

        try:
            c = Complex.from_pdb(
                pdb_fn,
                target_kwargs={"target_name": xtal_name},
                ligand_kwargs={"compound_name": compound_name},
            )
            return c
        except Exception as e:
            if fail_missing:
                raise ValueError(f"Unable to parse PDB file for {xtal_name}.") from e
            else:
                logger.warn(f"Unable to parse PDB file for {xtal_name}.")
                return None

    @classmethod
    def from_dir(
        cls,
        parent_dir: str | Path,
        metadata_csv_name="metadata.csv",
        xtal_col="crystal_name",
        compound_col="alternate_name",
        fail_missing=False,
    ) -> FragalysisFactory:
        """
        Build a FragalysisFactory from a Fragalysis directory.

        Parameters
        ----------
        parent_dir : str | Path
            Top-level directory of the Fragalysis database
        xtal_col : str, default="crystal_name"
            Name of the column in metadata csv giving the crystal names. Defaults to the
            Fragalysis value. The values in this col MUST match the directories in
            the aligned/ subdirectory
        compound_col : str, default="alternate_name"
            Name of the column in metadata csv giving the compound names. Defaults to
            the Fragalysis value
        fail_missing : bool, default=False
            If True, raises an error if a PDB file isn't found where expected, or a
            found PDB file can't be parsed

        Returns
        -------
        FragalysisFactory
        """
        return cls(
            parent_dir=Path(parent_dir),
            xtal_col=xtal_col,
            compound_col=compound_col,
            fail_missing=fail_missing,
            metadata_csv_name=metadata_csv_name,
        )
