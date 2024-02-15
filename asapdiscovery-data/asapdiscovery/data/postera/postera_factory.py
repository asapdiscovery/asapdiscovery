from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.schema_v2.ligand import Ligand, LigandIdentifiers
from asapdiscovery.data.services_config import PosteraSettings
from pydantic import BaseModel, Field, root_validator
from typing import Optional


class PosteraFactory(BaseModel):
    settings: PosteraSettings = Field(default_factory=PosteraSettings)
    molecule_set_name: Optional[str] = Field(
        None, description="Name of the molecule set to pull from Postera"
    )
    molecule_set_id: Optional[str] = Field(None, description="ID of the molecule set to pull from Postera"
    )

    @root_validator
    @classmethod
    def check_molecule_set_name_or_id(cls, values):
        molecule_set_name = values.get("molecule_set_name")
        molecule_set_id = values.get("molecule_set_id")
        if molecule_set_name is None and molecule_set_id is None:
            raise ValueError("Either molecule_set_name or molecule_set_id must be set")
        return values

    def pull(self) -> list[Ligand]:
        """
        Pull molecules from a Postera molecule set

        Returns
        -------
        List[Ligand]
            List of ligands
        """
        ms_api = MoleculeSetAPI.from_settings(self.settings)
        mols, _ = ms_api.get_molecules_from_id_or_name(name=self.molecule_set_name, id=self.molecule_set_id)
        ligands = [
            Ligand.from_smiles(
                compound_name=mol.id,
                smiles=mol.smiles,
                ids=LigandIdentifiers(manifold_api_id=mol.id),
            )
            for _, mol in mols.iterrows()
        ]
        return ligands
