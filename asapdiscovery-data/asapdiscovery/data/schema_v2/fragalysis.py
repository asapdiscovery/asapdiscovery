from pathlib import Path
from typing import Any, Union

import pandas
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.schema_base import (
    ContainerAbstractBase,
    DataModelAbstractBase,
)
from pydantic import Field


class FragalysisFactory(ContainerAbstractBase):
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
    complexes: List[Complex] = Field(
        [], description="Complex objects in the Fragalysis database."
    )

    @classmethod
    def from_dir(cls, parent_dir: str | Path):
        parent_dir = Path(parent_dir)
        try:
            df = pandas.read_csv(parent_dir / "metadata.csv")
        except FileNotFoundError as e:


    @root_validator(pre=True)
    @classmethod
    def _validate_parent_dir(cls, v):
        if not v.parent_dir.exists():
            raise ValueError("Given parent_dir does not exist.")

        if not v.parent_dir.is_dir():
            raise ValueError("Given parent_dir is not a directory.")
