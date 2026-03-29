import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.services.postera.postera_factory import PosteraFactory
from asapdiscovery.data.services.services_config import PosteraSettings

logger = logging.getLogger(__name__)


class MetaLigandFactory(BaseModel):
    """
    Factory for loading ligands from file or Postera

    Parameters
    ----------
    postera : bool
        use Postera
    postera_molset_name : str
        Postera molecule set name
    ligand_file : str
        Ligand file to read
    """

    postera: bool = Field(..., description="use Postera")
    postera_molset_name: Optional[str] = Field(
        ..., description="Postera molecule set name"
    )
    ligand_file: Optional[str | Path] = Field(..., description="Ligand file to read")

    @model_validator(mode="after")
    def options_mutex(self):
        if self.postera and self.ligand_file:
            raise ValueError("cannot specify postera and ligand_file")
        return self

    @model_validator(mode="after")
    def postera_molset_and_name(self):
        if self.postera and not self.postera_molset_name:
            raise ValueError("must specify postera_molset_name if postera is specified")
        return self

    def load(self) -> list[Ligand]:
        if self.postera:
            # load postera
            logger.info(
                f"Loading Postera database molecule set {self.postera_molset_name}"
            )
            postera_settings = PosteraSettings()
            postera = PosteraFactory(
                settings=postera_settings, molecule_set_name=self.postera_molset_name
            )
            query_ligands = postera.pull()
        else:
            # load from file
            logger.info(f"Loading ligands from file: {self.ligand_file}")
            molfile = MolFileFactory(filename=self.ligand_file)
            query_ligands = molfile.load()
        return query_ligands
