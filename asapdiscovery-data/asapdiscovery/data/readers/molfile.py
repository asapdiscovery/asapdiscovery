import logging
from pathlib import Path
from typing import Union

from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.schema.ligand import Ligand
from pydantic.v1 import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MolFileFactory(BaseModel):
    """
    Factory for a loading a generic molecule file into a list of Ligand objects.
    """

    filename: Union[str, Path] = Field(..., description="Path to the molecule file")

    def load(self) -> list[Ligand]:
        ifs = oechem.oemolistream()
        retcode = ifs.open(str(self.filename))
        if not retcode:
            raise ValueError(f"Could not open {self.filename}")

        ligands = []
        for i, mol in enumerate(ifs.GetOEGraphMols()):
            compound_name = mol.GetTitle()
            if not compound_name:
                compound_name = f"unknown_ligand_{i}"
            # can possibly do more here to get more information from the molecule
            # but for now just get the name, as the rest of the information is
            # not often stored in a consistent way eg in SD tags
            ligand = Ligand.from_oemol(mol, compound_name=compound_name)
            ligands.append(ligand)
        return ligands

    @validator("filename")
    @classmethod
    def check_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"File {v} does not exist")
        return v
