from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.schema_v2.ligand import Ligand, LigandIdentifiers
from asapdiscovery.data.services_config import PosteraSettings
from pydantic import BaseModel, Field


class PosteraFactory(BaseModel):
    settings: PosteraSettings = Field(default_factory=PosteraSettings)
    molecule_set_name: str = Field(
        ..., description="Name of the molecule set to pull from Postera"
    )

    def pull(self) -> list[Ligand]:
        """
        Pull molecules from a Postera molecule set

        Returns
        -------
        List[Ligand]
            List of ligands
        """
        ms_api = MoleculeSetAPI.from_settings(self.settings)
        mols, _ = ms_api.get_molecules_from_id_or_name(self.molecule_set_name)
        ligands = [
            Ligand.from_smiles(
                smiles=mol.smiles, ids=LigandIdentifiers(manifold_api_id=mol.id)
            )
            for _, mol in mols.iterrows()
        ]
        return ligands
