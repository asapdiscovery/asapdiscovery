from uuid import UUID

from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.postera.manifold_data_validation import ManifoldAllowedTags
from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI, MoleculeSetKeys
from asapdiscovery.data.rdkit import rdkit_smiles_roundtrip
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
    overwrite: bool = Field(
        False, description="Overwrite existing data on molecule set"
    )

    def push(self, df) -> None:
        """
        Push molecules to a Postera molecule set

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload

        Returns
        -------
        DataFrame
            The dataframe with the manifold id's added if a new molecule set was created
        molecule_set_id : UUID
            The UUID of the molecule set
        new_molset : bool
        """
        ms_api = MoleculeSetAPI.from_settings(self.settings)
        df_copy = df.copy()
        new_molset = False
        molset_name = self.molecule_set_name
        if not ms_api.exists(self.molecule_set_name, by="name"):
            id = ms_api.create_molecule_set_from_df_with_manifold_validation(
                molecule_set_name=self.molecule_set_name,
                df=df,
                id_field=self.id_field,
                smiles_field=self.smiles_field,
            )
            new_data = ms_api.get_molecules(id, return_as="dataframe")
            df_copy = self.join_with_manifold_data(df_copy, new_data)
            new_molset = True
        else:
            molset_id = ms_api.molecule_set_id_or_name(
                self.molecule_set_name, ms_api.list_available()
            )

            if not self.id_data_is_uuid_castable(df, self.id_field):
                new_data = ms_api.get_molecules(molset_id, return_as="dataframe")
                df_copy = self.join_with_manifold_data(df_copy, new_data)

            ms_api.update_molecules_from_df_with_manifold_validation(
                molecule_set_id=molset_id,
                df=df_copy,
                id_field=self.id_field,
                smiles_field=self.smiles_field,
                overwrite=self.overwrite,
            )
        return df_copy, molset_name, new_molset

    @staticmethod
    def join_with_manifold_data(original, molset_query_df):
        """
        Join the original dataframe with manifold data
        that is returned from a query to the manifold API
        """
        data = original.copy()
        subset = molset_query_df[
            [MoleculeSetKeys.id.value, MoleculeSetKeys.smiles.value]
        ]

        # use rdkit here, to match postera backend which uses rdkit
        # provides better matching performance with smiles pulled down from postera

        # do a roundtrip to canonicalize the smiles
        subset.loc[:, MoleculeSetKeys.smiles.value] = subset[
            MoleculeSetKeys.smiles.value
        ].apply(rdkit_smiles_roundtrip)

        # do the same to the original data
        data.loc[:, ManifoldAllowedTags.SMILES.value] = data[
            ManifoldAllowedTags.SMILES.value
        ].apply(rdkit_smiles_roundtrip)

        # give it the right column names
        subset.rename(
            columns={MoleculeSetKeys.smiles.value: ManifoldAllowedTags.SMILES.value},
            inplace=True,
        )
        # merge the data
        data = data.merge(subset, on=ManifoldAllowedTags.SMILES.value, how="inner")

        # drop original ID column and replace with the manifold ID
        data.drop(columns=[DockingResultCols.LIGAND_ID.value], inplace=True)
        data.rename(
            columns={MoleculeSetKeys.id.value: DockingResultCols.LIGAND_ID.value},
            inplace=True,
        )
        return data

    @staticmethod
    def id_data_is_uuid_castable(df, id_field) -> bool:
        """
        Check if the id data is castable to UUID

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload
        id_field : str
            Name of the column in the dataframe to use as the ligand id

        Returns
        -------
        bool
            Whether the data is castable to UUID
        """
        try:
            df[id_field].apply(lambda x: UUID(x))
            return True
        except ValueError:
            return False
