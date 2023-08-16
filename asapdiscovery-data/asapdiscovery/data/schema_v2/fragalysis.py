from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from pydantic import Field, validator


class FragalysisFactory(DataModelAbstractBase):
    """
    Schema for a loading a Fragalysis dump. The directoory given by parent_dir should
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

    # Overload from base class to check each complex
    def data_equal(self, other: FragalysisFactory):
        return all(
            [c1.data_equal(c2) for c1, c2 in zip(self.complexes, other.complexes)]
        )

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
        parent_dir = Path(parent_dir)
        try:
            df = pandas.read_csv(parent_dir / "metadata.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError("No metadata.csv file found in parent_dir.") from e

        if len(df) == 0:
            raise ValueError("metadata.csv file is empty.")

        if (xtal_col not in df.columns) or (compound_col not in df.columns):
            raise ValueError(
                "metadata.csv file must contain a crystal name column and a "
                "compound name column."
            )

        # Dict mapping crystal name to compound name
        xtal_compound_dict = dict(zip(df[xtal_col], df[compound_col]))

        try:
            all_xtal_dirs = os.listdir(parent_dir / "aligned")
        except FileNotFoundError as e:
            raise FileNotFoundError("No aligned/ directory found in parent_dir.") from e

        all_xtal_dirs = [d for d in all_xtal_dirs if d in xtal_compound_dict]
        if len(all_xtal_dirs) == 0:
            raise ValueError(
                "No aligned directories found with entries in metadata.csv"
            )

        # Loop through directories and load each bound file
        complexes = []
        for d in all_xtal_dirs:
            compound_name = xtal_compound_dict[d]
            pdb_fn = parent_dir / "aligned" / d / f"{d}_bound.pdb"
            if not pdb_fn.exists():
                if fail_missing:
                    raise FileNotFoundError(f"No PDB file found for {d}.")
                else:
                    print(f"No PDB file found for {d}.", flush=True)
                    continue

            try:
                c = Complex.from_pdb(
                    pdb_fn,
                    target_kwargs={"target_name": d},
                    ligand_kwargs={"compound_name": compound_name},
                )
            except Exception as e:
                if fail_missing:
                    raise ValueError(f"Unable to parse PDB file for {d}.") from e
                else:
                    print(f"Unable to parse PDB file for {d}.", flush=True)
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
