import logging
from pathlib import Path
from typing import Optional

from asapdiscovery.data.readers.structure_dir import StructureDirFactory
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.services.fragalysis.fragalysis_reader import FragalysisFactory
from asapdiscovery.data.util.dask_utils import FailureMode
from pydantic.v1 import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)


class MetaStructureFactory(BaseModel):
    """
    Factory for loading structures from directory, fragalysis or pdb file

    Parameters
    ----------
    structure_dir : str
        Path to directory containing structures
    fragalysis_dir : str
        Path to directory containing fragalysis structures
    pdb_file : str
        Path to pdb file containing structure
    use_dask : bool
        Use dask to load structures where possible
    dask_client : Any
        Dask client to use for loading structures
    """

    structure_dir: Optional[str | Path] = Field(
        ..., description="Path to directory containing structures"
    )
    fragalysis_dir: Optional[str | Path] = Field(
        ..., description="Path to directory containing fragalysis structures"
    )
    pdb_file: Optional[str | Path] = Field(
        ..., description="Path to pdb file containing structure"
    )

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

    def load(
        self,
        use_dask: bool = False,
        dask_client=None,
        failure_mode: FailureMode = FailureMode.SKIP,
    ) -> list[Complex]:
        # load complexes from a directory, from fragalysis or from a pdb file
        if self.structure_dir:
            logger.info(f"Loading structures from directory: {self.structure_dir}")
            structure_factory = StructureDirFactory.from_dir(self.structure_dir)
            complexes = structure_factory.load(
                use_dask=use_dask,
                dask_client=dask_client,
                failure_mode=failure_mode,
            )
        elif self.fragalysis_dir:
            logger.info(f"Loading structures from fragalysis: {self.fragalysis_dir}")
            fragalysis = FragalysisFactory.from_dir(self.fragalysis_dir)
            complexes = fragalysis.load(
                use_dask=use_dask,
                dask_client=dask_client,
                failure_mode=failure_mode,
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
