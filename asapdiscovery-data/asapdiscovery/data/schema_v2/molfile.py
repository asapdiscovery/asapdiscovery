from pathlib import Path
from typing import List  # noqa: F401

from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from pydantic import Field, validator


class MolFileFactory(DataModelAbstractBase):
    """
    Factory for a loading a generic molecule file into a list of Ligand objects.
    """

    filename: str = Field(..., description="Path to the molecule file")
    ligands: list[Ligand] = Field(..., description="List of Ligand objects")

    @classmethod
    def from_file(cls, filename) -> list[Ligand]:
        filename = str(filename)

        ifs = oechem.oemolistream()
        retcode = ifs.open(str(filename))
        if not retcode:
            raise ValueError(f"Could not open {filename}")

        mollist = []
        for mol in ifs.GetOEGraphMols():
            mollist.append(oechem.OEGraphMol(mol))

        ligands = []
        for i, mol in enumerate(mollist):
            compound_name = mol.GetTitle()
            if not compound_name:
                compound_name = f"unknown_ligand_{i}"
            # can possibly do more here to get more information from the molecule
            # but for now just get the name, as the rest of the information is
            # not often stored in a consistent way eg in SDF tags
            ligand = Ligand.from_oemol(mol, compound_name=compound_name)
            ligands.append(ligand)

        return cls(filename=filename, ligands=ligands)

    @validator("filename")
    @classmethod
    def check_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"File {v} does not exist")
        return v
