from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.services_config import PosteraSettings
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from pydantic import BaseModel, Field


class PosteraUploader(BaseModel):
    settings: PosteraSettings = Field(default_factory=PosteraSettings)
    molecule_set_name: str = Field(
        ...,
        description="Name of the molecule set to push to Postera, if it doesn't exist it will be created",
    )
    id_field: str = Field(
        DockingResultCols.LIGAND_ID.value,
        description="Name of the column in the dataframe to use as the ligand id",
    )
    smiles_field: str = Field(
        DockingResultCols.SMILES.value,
        description="Name of the column in the dataframe to use as the SMILES field",
    )

    def push(self, df) -> None:
        """
        Push molecules to a Postera molecule set

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload
        """
        ms_api = MoleculeSetAPI.from_settings(self.settings)
        if not ms_api.exists(self.molecule_set_name, by="name"):
            ms_api.create_molecule_set_from_df_with_manifold_validation(
                molecule_set_name=self.molecule_set_name,
                df=df,
                id_field=self.id_field,
                smiles_field=self.smiles_field,
            )
        else:
            molset_id = ms_api.molecule_set_id_or_name(
                self.molecule_set_name, ms_api.list_available()
            )

            ms_api.update_molecules_from_df_with_manifold_validation(
                molecule_set_id=molset_id,
                df=df,
                id_field=self.id_field,
                smiles_field=self.smiles_field,
                overwrite=True,
            )
