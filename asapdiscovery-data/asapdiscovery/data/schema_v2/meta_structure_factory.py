import logging
from pathlib import Path
from typing import Any

from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.fragalysis import FragalysisFactory
from asapdiscovery.data.schema_v2.structure_dir import StructureDirFactory
from pydantic import BaseModel, Field, root_validator, validator

logger = logging.getLogger(__name__)


class MetaStructureFactory(BaseModel):
    structure_dir: str | Path = Field(
        ..., description="Path to directory containing structures"
    )
    fragalysis_dir: str | Path = Field(
        ..., description="Path to directory containing fragalysis structures"
    )
    pdb_file: str | Path = Field(
        ..., description="Path to pdb file containing structure"
    )
    use_dask: bool = Field(False, description="Use dask to load structures")
    dask_client: Any = Field(
        None, description="Dask client to use for loading structures"
    )

    # convert strings to Path objects
    @validator("structure_dir", "fragalysis_dir", "pdb_file", pre=True)
    def string_to_path(cls, v):
        return Path(v)

    @root_validator
    def options_mutex(cls, values):
        fragalysis = values.get("fragalysis_dir")
        pdb_file = values.get("pdb_file")
        structure_dir = values.get("structure_dir")
        vals = [fragalysis, pdb_file, structure_dir]
        if sum(bool(v) for v in vals) != 1:
            raise ValueError(
                "Must specify exactly one of structure_dir, fragalysis_dir or pdb_file"
            )
        return values

    def load(self) -> list[Complex]:
        # load complexes from a directory, from fragalysis or from a pdb file
        if self.structure_dir:
            logger.info(f"Loading structures from directory: {self.structure_dir}")
            structure_factory = StructureDirFactory.from_dir(self.structure_dir)
            complexes = structure_factory.load(
                use_dask=self.use_dask, dask_client=self.dask_client
            )
        elif self.fragalysis_dir:
            logger.info(f"Loading structures from fragalysis: {self.fragalysis_dir}")
            fragalysis = FragalysisFactory.from_dir(self.fragalysis_dir)
            complexes = fragalysis.load(
                use_dask=self.use_dask, dask_client=self.dask_client
            )

        elif self.pdb_file:
            logger.info(f"Loading structures from pdb: {self.pdb_file}")
            complex = Complex.from_pdb(
                self.pdb_file,
                target_kwargs={"target_name": self.pdb_file.stem},
                ligand_kwargs={"compound_name": f"{self.pdb_file.stem}_ligand"},
            )
            complexes = [complex]

        return complexes
