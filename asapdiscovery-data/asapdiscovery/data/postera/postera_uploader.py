import logging
import warnings
from uuid import UUID
import pandas as pd
import numpy as np
from typing import Tuple

from asapdiscovery.data.postera.manifold_data_validation import ManifoldAllowedTags
from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI, MoleculeSetKeys
from asapdiscovery.data.rdkit import rdkit_smiles_roundtrip
from asapdiscovery.data.services_config import PosteraSettings
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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

    def push(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, UUID, bool]:
        """
        Push molecules to a Postera molecule set

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload

        Returns
        -------
        DataFrame
            The input dataframe merged with the data from the molecule set, including UUIDs
        molecule_set_id : UUID
            The UUID of the molecule set
        new_molset : bool
            Whether a new molecule set was created
        """

        if not self.smiles_field in df.columns:
            raise ValueError(f"smiles_field {self.smiles_field} not found in dataframe")
        if not self.id_field in df.columns:
            raise ValueError(f"id_field {self.id_field} not found in dataframe")

        ms_api = MoleculeSetAPI.from_settings(self.settings)
        data = df.copy()
        new_molset = False

        # if the molecule set doesn't exist, create it
        if not ms_api.exists(self.molecule_set_name, by="name"):
            logger.debug(
                f"molecule set {self.molecule_set_name} does not exist, creating new molecule set"
            )
            molset_id = ms_api.create_molecule_set_from_df_with_manifold_validation(
                molecule_set_name=self.molecule_set_name,
                df=df,
                id_field=self.id_field,
                smiles_field=self.smiles_field,
            )
            # get the new data including manifold UUIDs and join it with the original data
            new_molset = True

        else:
            # grab id of molecule set
            molset_id = ms_api.get_id_from_name(self.molecule_set_name)
            logger.debug(
                f"molecule set {self.molecule_set_name} exists with id {molset_id}, updating molecule set"
            )

            if not self.id_data_is_uuid_castable(data, self.id_field):
                # we need to get the data from the manifold API and join it with the original data
                new_data = ms_api.get_molecules(molset_id, return_as="dataframe")
                data = self.join_with_manifold_data(
                    data,
                    new_data,
                    smiles_field=self.smiles_field,
                    id_field=self.id_field,
                )
                self._check_for_duplicates(
                    data, self.id_field, allow_empty=True, raise_error=True
                )

                # find rows with blank id, they need to be added to molset, using **add** endpoint rather than **update**
                has_blank_id_rows, blank_id_rows = self._check_for_blank_ids(
                    data, self.id_field, raise_error=False
                )
                logger.debug(f"data has_blank_id_rows: {has_blank_id_rows}")
                if has_blank_id_rows:
                    ms_api.add_molecules_from_df_with_manifold_validation(
                        molecule_set_id=molset_id,
                        df=blank_id_rows,
                        id_field=self.id_field,
                        smiles_field=self.smiles_field,
                    )
                    logger.debug(
                        "appending to molecule set where some molecules have not been matched to an existing molecule in the molecule set, these ligands will be added to the molecule set"
                    )

                # find rows with a UUID, they need to be updated using the **update** endpoint
                uuid_rows = data[~data[self.id_field].isna()]

                ms_api.update_molecules_from_df_with_manifold_validation(
                    molecule_set_id=molset_id,
                    df=uuid_rows,
                    id_field=self.id_field,
                    smiles_field=self.smiles_field,
                    overwrite=self.overwrite,
                )

            else:
                # if the id data is castable to UUID, we can just update the molecule set

                # check for duplicates
                self._check_for_duplicates(
                    data, self.id_field, allow_empty=False, raise_error=True
                )
                # check for blanks, raising
                self._check_for_blank_ids(data, self.id_field, raise_error=True)

                # ok to update the molecule set
                ms_api.update_molecules_from_df_with_manifold_validation(
                    molecule_set_id=molset_id,
                    df=data,
                    id_field=self.id_field,
                    smiles_field=self.smiles_field,
                    overwrite=self.overwrite,
                )

        new_data = ms_api.get_molecules(molset_id, return_as="dataframe")
        molset_name = ms_api.get_name_from_id(molset_id)
        return new_data, molset_name, new_molset

    @staticmethod
    def join_with_manifold_data(original, molset_query_df, smiles_field, id_field):
        """
        Join the original dataframe with manifold data
        that is returned from a query to the manifold API


        Parameters
        ----------
        original : DataFrame
            The original dataframe
        molset_query_df : DataFrame
            The dataframe returned from a query to the manifold API
        smiles_field : str
            The name of the smiles field in the original dataframe
        id_field : str
            The name of the id field in the original dataframe
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
        data.loc[:, smiles_field] = data[smiles_field].apply(rdkit_smiles_roundtrip)

        # rename
        subset.rename(
            columns={MoleculeSetKeys.smiles.value: smiles_field},
            inplace=True,
        )

        # merge the data, outer join very important here to avoid dropping rows that are present in local data but not in manifold
        data = data.merge(subset, on=smiles_field, how="outer", suffixes=("", "_y"))
        data.drop(data.filter(regex="_y$").columns, axis=1, inplace=True)
        # drop original ID column and replace with the manifold ID
        if id_field != MoleculeSetKeys.id.value:
            data.drop(columns=id_field, inplace=True)
            data.rename(
                columns={MoleculeSetKeys.id.value: id_field},
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
            Whether the entire data column is castable to UUID
        """
        try:
            df[id_field].apply(lambda x: UUID(x))
            return True
        except:
            return False

    @staticmethod
    def _check_for_duplicates(df, id_field, allow_empty=True, raise_error=False):
        """
        Check for duplicate UUIDs in the dataframe

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload
        id_field : str
            Name of the column in the dataframe to use as the ligand id
        allow_empty : bool
            Whether to allow empty UUIDs to be exempt from the check
        raise_error : bool
            Whether to raise an error if duplicates are found

        Raises
        ------
        ValueError
            If there are duplicate UUIDs
        """
        df = df.copy()
        df = df.replace("", np.nan)
        if allow_empty:
            df = df[~df[id_field].isna()]
        if df[id_field].duplicated().any():
            duplicates = df[df[id_field].duplicated()]
            num_duplicates = len(duplicates)
            if raise_error:
                raise ValueError(f"{num_duplicates} duplicate UUIDs found in dataframe")
            return True, duplicates
        else:
            return False, None

    @staticmethod
    def _check_for_blank_ids(df, id_field, raise_error=False):
        """
        Check for blank UUIDs in the dataframe

        Parameters
        ----------
        df : DataFrame
            DataFrame of data to upload
        id_field : str
            Name of the column in the dataframe to use as the ligand id

        Raises
        ------
        ValueError
            If there are blank UUIDs
        """
        df = df.copy()
        df = df.replace("", np.nan)
        if df[id_field].isna().any():
            if raise_error:
                raise ValueError("Blank UUIDs found in dataframe")
            return True, df[df[id_field].isna()]

        else:
            return False, None
