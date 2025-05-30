import logging
from pathlib import Path
from typing import List  # noqa: F401

import dask
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.util.dask_utils import (
    FailureMode,
    actualise_dask_delayed_iterable,
)
from pydantic.v1 import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class StructureDirFactory(BaseModel):
    """
    Factory for loading a directory of PDB files as Complex objects.

    Parameters
    ----------
    parent_dir : str or Path
        Directory containing PDB files.
    """

    parent_dir: Path = Field(
        description="Directory containing structure files as PDBs."
    )
    glob: str = Field(
        default="*.pdb",
        description="Regex pattern for matching PDB files in the directory.",
    )

    @validator("parent_dir")
    def parent_dir_exists(cls, v):
        if not v.exists():
            raise ValueError("parent_dir does not exist.")
        return v

    @classmethod
    def from_dir(cls, parent_dir: Path | str):
        """
        Load a directory of PDB files as Complex objects.
        """
        return cls(parent_dir=Path(parent_dir))

    def load(self, use_dask=True, dask_client=None, failure_mode=FailureMode.SKIP):
        """
        Load a directory of PDB files as Complex objects.

        Parameters
        ----------
        use_dask : bool, optional
            Whether to use dask to parallelise loading of PDB files.
            Defaults to True.
        dask_client : dask.distributed.Client, optional
            Dask client to use for parallelisation. Defaults to None.
        failure_mode : FailureMode
            The failure mode for dask. Can be 'raise' or 'skip'.

        Returns
        -------
        List[Complex]
            List of Complex objects.
        """
        pdb_files = list(self.parent_dir.glob(self.glob))
        # check all filenames are unique
        pdb_stems = [pdb_file.stem for pdb_file in pdb_files]
        unique = False
        if len(pdb_stems) == len(set(pdb_stems)):
            unique = True

        if use_dask:
            delayed_outputs = []
            for i, pdb_file in enumerate(pdb_files):
                stem = pdb_file.stem
                if not unique:
                    stem = f"{stem}_{i}"
                out = dask.delayed(Complex.from_pdb)(
                    pdb_file,
                    target_kwargs={"target_name": stem},
                    ligand_kwargs={"compound_name": f"{stem}_ligand"},
                )
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors=failure_mode
            )
        else:
            outputs = []
            for i, pdb_file in enumerate(pdb_files):
                stem = pdb_file.stem
                if not unique:
                    stem = f"{stem}_{i}"
                out = Complex.from_pdb(
                    pdb_file,
                    target_kwargs={"target_name": stem},
                    ligand_kwargs={"compound_name": f"{stem}_ligand"},
                )
                outputs.append(out)

        return outputs
