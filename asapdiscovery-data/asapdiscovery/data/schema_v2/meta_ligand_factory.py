import logging
from pathlib import Path

from asapdiscovery.data.postera.postera_factory import PosteraFactory
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.molfile import MolFileFactory
from asapdiscovery.data.services_config import PosteraSettings
from pydantic import BaseModel, Field, root_validator


logger = logging.getLogger(__name__)


class MetaLigandFactory(BaseModel):
    postera: bool = Field(..., description="use Postera")
    postera_molset_name: str = Field(..., description="Postera molecule set name")
    ligand_file: str | Path = Field(..., description="Ligand file to read")

    @root_validator
    @classmethod
    def options_mutex(cls, values):
        postera = values.get("postera")
        ligand_file = values.get("ligand_file")
        if postera and ligand_file:
            raise ValueError("cannot specify postera and ligand_file")
        return values

    @root_validator
    @classmethod
    def postera_molset_and_name(cls, values):
        postera_molset_name = values.get("postera_molset_name")
        postera = values.get("postera")
        if not (postera and postera_molset_name):
            raise ValueError("must specify postera_molset_name if postera is specified")
        return values

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
            molfile = MolFileFactory.from_file(self.ligand_file)
            query_ligands = molfile.ligands

        return query_ligands
