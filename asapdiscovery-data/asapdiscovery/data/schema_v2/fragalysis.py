from __future__ import annotations

import os
from pathlib import Path
from typing import List  # noqa: F401

import pandas
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
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
    complexes: list[Complex] = Field(
        [], description="Complex objects in the Fragalysis database.", repr=False
    )

    def __len__(self):
        return len(self.complexes)

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

    @classmethod
    def from_dir(
        cls,
        parent_dir: str | Path,
        csv_name="metadata.csv",
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
        parent_dir = Path(parent_dir)
        try:
            df = pandas.read_csv(parent_dir / csv_name)
        except FileNotFoundError as e:
            raise FileNotFoundError("No metadata.csv file found in parent_dir.") from e

        if len(df) == 0:
            raise ValueError("metadata.csv file is empty.")

        if (xtal_col not in df.columns) or (compound_col not in df.columns):
            raise ValueError(
                "metadata.csv file must contain a crystal name column and a "
                "compound name column."
            )

        try:
            all_xtal_dirs = os.listdir(parent_dir / "aligned")
        except FileNotFoundError as e:
            raise FileNotFoundError("No aligned/ directory found in parent_dir.") from e

        # Subset metadata to only contain rows with directories in aligned/
        df = df.loc[df[xtal_col].isin(all_xtal_dirs), :]
        if df.shape[0] == 0:
            raise ValueError(
                "No aligned directories found with entries in metadata.csv."
            )

        # Loop through directories and load each bound file
        complexes = []
        for _, (xtal_name, compound_name) in df[[xtal_col, compound_col]].iterrows():
            pdb_fn = parent_dir / "aligned" / xtal_name / f"{xtal_name}_bound.pdb"
            if not pdb_fn.exists():
                if fail_missing:
                    raise FileNotFoundError(f"No PDB file found for {xtal_name}.")
                else:
                    print(f"No PDB file found for {xtal_name}.", flush=True)
                    continue

            try:
                c = Complex.from_pdb(
                    pdb_fn,
                    target_kwargs={"target_name": xtal_name},
                    ligand_kwargs={"compound_name": compound_name},
                )
            except Exception as e:
                if fail_missing:
                    raise ValueError(
                        f"Unable to parse PDB file for {xtal_name}."
                    ) from e
                else:
                    print(f"Unable to parse PDB file for {xtal_name}.", flush=True)
                    continue

            complexes.append(c)

        return cls(parent_dir=parent_dir, complexes=complexes)

    @validator("parent_dir")
    @classmethod
    def _validate_parent_dir(cls, v):
        if not v.exists():
            raise ValueError("Given parent_dir does not exist.")

        if not v.is_dir():
            raise ValueError("Given parent_dir is not a directory.")

        return v
