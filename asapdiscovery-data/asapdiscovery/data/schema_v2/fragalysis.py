from __future__ import annotations

import dask
import os
from pathlib import Path
from typing import List  # noqa: F401
import warnings
import pandas
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from pydantic import Field, validator


class FragalysisFactory(DataModelAbstractBase):
    """
    Schema for a loading a Fragalysis dump. The directory given by parent_dir should
    contain (at a minimum):
     * a metadata.csv file with the columns "crystal_name" and "alternate_name"
     * an aligned/ subdirectory, containing a subdirectory for each entry in
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

    def __eq__(self, other: FragalysisFactory):
        if self.parent_dir != other.parent_dir:
            return False

        if len(self) != len(other):
            return False

        return all([c1 == c2 for c1, c2 in zip(self.complexes, other.complexes)])

    # Overload from base class to check each complex
    def data_equal(self, other: FragalysisFactory):
        if len(self) != len(other):
            return False

        return all(
            [c1.data_equal(c2) for c1, c2 in zip(self.complexes, other.complexes)]
        )

    def load(self, use_dask=False, dask_client=None) -> List[Complex]:
        try:
            df = pandas.read_csv(self.parent_dir / "metadata.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError("No metadata.csv file found in parent_dir.") from e

        if len(df) == 0:
            raise ValueError("metadata.csv file is empty.")

        if (self.xtal_col not in df.columns) or (self.compound_col not in df.columns):
            raise ValueError(
                "metadata.csv file must contain a crystal name column and a "
                "compound name column."
            )

        try:
            all_xtal_dirs = os.listdir(self.parent_dir / "aligned")
        except FileNotFoundError as e:
            raise FileNotFoundError("No aligned/ directory found in parent_dir.") from e

        # Subset metadata to only contain rows with directories in aligned/
        df = df.loc[df[self.xtal_col].isin(all_xtal_dirs), :]
        if df.shape[0] == 0:
            raise ValueError(
                "No aligned directories found with entries in metadata.csv."
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
                complexes, dask_client=dask_client
            )

        # remove None values
        complexes = [c for c in complexes if c is not None]

        return complexes

    @staticmethod
    def process_fragalysis_pdb(
        parent_dir, xtal_name, compound_name, fail_missing=False
    ):
        pdb_fn = parent_dir / "aligned" / xtal_name / f"{xtal_name}_bound.pdb"
        if not pdb_fn.exists():
            if fail_missing:
                raise FileNotFoundError(f"No PDB file found for {xtal_name}.")
            else:
                warnings.warn(f"No PDB file found for {xtal_name}.")
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
                print(f"Unable to parse PDB file for {xtal_name}.", flush=True)
                return None

    @classmethod
    def from_dir(
        cls,
        parent_dir: str | Path,
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
            Name of the column in metadata.csv giving the crystal names. Defaults to the
            Fragalysis value. The values in this col MUST match the directories in
            the aligned/ subdirectory
        compound_col : str, default="alternate_name"
            Name of the column in metadata.csv giving the compound names. Defaults to
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
        )

    @validator("parent_dir")
    @classmethod
    def _validate_parent_dir(cls, v):
        if not v.exists():
            raise ValueError("Given parent_dir does not exist.")

        if not v.is_dir():
            raise ValueError("Given parent_dir is not a directory.")

        return v
