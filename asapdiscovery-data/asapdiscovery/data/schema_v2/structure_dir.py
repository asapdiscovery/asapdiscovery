import os
import warnings
from pathlib import Path
from typing import List  # noqa: F401

import dask
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from pydantic import BaseModel, Field, validator


class StructureDirFactory(BaseModel):
    parent_dir: Path = Field(
        description="Top level directory of the Fragalysis database."
    )

    @validator("parent_dir")
    def parent_dir_exists(cls, v):
        if not v.exists():
            raise ValueError("parent_dir does not exist.")
        return v

    @classmethod
    def from_dir(cls, parent_dir: Path | str):
        return cls(parent_dir=Path(parent_dir))

    def load(self, use_dask=True, dask_client=None):
        pdb_files = list(self.parent_dir.glob("*.pdb"))
        if use_dask:
            delayed_outputs = []
            for pdb_file in pdb_files:
                out = dask.delayed(Complex.from_pdb)(pdb_file)
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors="raise"
            )
        else:
            outputs = [Complex.from_pdb(pdb_file) for pdb_file in pdb_files]

        return outputs
