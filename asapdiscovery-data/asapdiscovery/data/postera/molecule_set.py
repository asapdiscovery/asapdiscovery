import uuid
import warnings
from typing import Dict, Tuple, Union  # noqa: F401

import pandas as pd
from asapdiscovery.data.enum import StringEnum
from typing_extensions import TypedDict

from .manifold_data_validation import ManifoldAllowedTags
from .postera_api import PostEraAPI

_POSTERA_COLUMN_BLEACHING_ACTIVE = (
    True  # NOTE: this is fix for postera bleaching see issues #629 #628
)


class MoleculeSetKeys(StringEnum):
    """Keys for the response from the PostEra API when creating, getting or modifying a molecule set"""

    id = "id"
    smiles = "smiles"


class Molecule(TypedDict):
    """Data type to build MoleculeList"""

    smiles: str
    customData: dict[str, Union[str, float, int]]


class MoleculeUpdate(TypedDict):
    """Data type to build MoleculeUpdateList"""

    id: str
    customData: dict[str, Union[str, float, int]]


class MoleculeList(list[Molecule]):
    """Data type to pass to PostEra API in molecule set create"""

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
        smiles_field: str = MoleculeSetKeys.smiles.value,
        id_field: str = MoleculeSetKeys.id.value,
    ):
        return cls(
            [
                {
                    MoleculeSetKeys.smiles.value: row[smiles_field],
                    "customData": {
                        **{
                            key: value
                            for key, value in row.items()
                            if key not in [smiles_field, id_field]
                        },
                    },
                }
                for _, row in df.iterrows()
            ]
        )


class MoleculeUpdateList(list[MoleculeUpdate]):
    """Data type to pass to PostEra API in molecule set update_custom_data"""

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
        smiles_field: str = MoleculeSetKeys.smiles.value,
        id_field: str = MoleculeSetKeys.id.value,
    ):
        return cls(
            [
                {
                    MoleculeSetKeys.id.value: str(row[id_field]),
                    "customData": {
                        **{
                            key: value
                            for key, value in row.items()
                            if key not in [smiles_field, id_field]
                        },
                    },
                }
                for _, row in df.iterrows()
            ]
        )


class MoleculeSetAPI(PostEraAPI):
    """Connection and commands for PostEra Molecule Set API"""

    @classmethod
    def from_settings(cls, settings):
        """
        Create an interface to PostEra Molecule Set API from a `Settings` object.

        Parameters
        ----------
        settings
            A `PosteraSettings` object

        Returns
        -------
        MoleculeSetAPI
            MoleculeSetAPI interface object.
        """
        return cls(
            api_key=settings.POSTERA_API_KEY,
            url=settings.POSTERA_API_URL,
            api_version=settings.POSTERA_API_VERSION,
        )

    @property
    def molecule_set_url(self):
        return f"{self.api_url}/moleculesets"

    @staticmethod
    def molecule_set_id_or_name(
        id_or_name: str, available_molsets: dict[str, str]
    ) -> str:
        """
        Helper function to determine if the input is a molecule set id or name
        and return the molecule set id
        """
        try:
            uuid.UUID(id_or_name)
            molset_id = id_or_name
        except ValueError:
            available_molsets_rev = {v: k for k, v in available_molsets.items()}
            try:
                molset_id = available_molsets_rev[id_or_name]
            except KeyError:
                raise ValueError(
                    f"Molecule Set with identifier: {id_or_name} not found in PostEra"
                )
        return molset_id

    def create(
        self, molecule_set_name: str, data: MoleculeList, return_full: bool = False
    ) -> str:
        """Create a MoleculeSet from a list of Molecules.

        Parameters
        ----------
        set_name
            The human-readable name for the set.
        data
            MoleculeList giving Molecules to add.
        return_full
            If `True`, return a dict containing summary data for the created
            MoleculeSet; if `False`, return only its unique id.

        """
        url = f"{self.molecule_set_url}/"
        response = self._session.post(
            url,
            json={
                "molecules": data,
                "name": molecule_set_name,
            },
        ).json()

        if return_full:
            return response
        else:
            return response[MoleculeSetKeys.id.value]

    def _read_page(self, url: str, page: int) -> (pd.DataFrame, str):
        response = self._session.get(url, params={"page": page}).json()

        return response["results"], response["paginationInfo"]["hasNext"]

    def _collate(self, url):
        page = 0
        has_next = True
        results = []

        while has_next:
            page += 1
            result, has_next = self._read_page(url, page)
            results.extend(result)

        return results

    def list_available(self, return_full: bool = False) -> Union[list[dict], dict]:
        """List available MoleculeSets.

        Parameters
        ----------
        return_full
            If `True`, return a list of dicts containing summary data for each
            MoleculeSet; if `False`, return a dict with the unique id for each
            MoleculeSet as keys, human-readable name as values.

        """
        url = f"{self.molecule_set_url}/"

        results = self._collate(url)

        if return_full:
            return results
        else:
            return {
                result[MoleculeSetKeys.id.value]: result["name"] for result in results
            }

    def exists(self, molecule_set_name: str, by="name") -> bool:
        """
        Check if a molecule set exists in PostEra.

        Parameters
        ----------
        molecule_set_name
            The name of the molecule set to check.
        by
            The identifier type to check by. Can be either "id" or "name".

        Returns
        -------
        bool
            Whether the molecule set exists in PostEra.
        """
        avail = self.list_available()
        if by == "id":
            return molecule_set_name in avail.keys()
        elif by == "name":
            return molecule_set_name in avail.values()
        else:
            raise ValueError(f"Unknown identifier type: {by}")

    def get(self, molecule_set_id: str) -> dict:
        """Get summary data for a given MoleculeSet.

        Parameters
        ----------
        molecule_set_id
            The unique id of the MoleculeSet

        Returns
        -------
        Summary data as a dict.

        """
        url = f"{self.molecule_set_url}/{molecule_set_id}"
        response = self._session.get(
            url,
        ).json()

        return response

    def get_molecules(
        self, molecule_set_id: str, return_as="dataframe"
    ) -> Union[pd.DataFrame, list]:
        """Pull the full contents of a MoleculeSet as a DataFrame.

        Parameters
        ----------
        molecule_set_id
            The unique id of the MoleculeSet
        return_as : {'dataframe', 'list'}

        """

        if return_as not in ("dataframe", "list"):
            raise ValueError("`return_as` must be either 'dataframe' or 'list'")

        url = f"{self.molecule_set_url}/{molecule_set_id}/get_all_molecules/"

        results = self._collate(url)

        if return_as == "list":
            return results
        elif return_as == "dataframe":
            response_data = [
                {
                    MoleculeSetKeys.smiles.value: result[MoleculeSetKeys.smiles.value],
                    MoleculeSetKeys.id.value: result[MoleculeSetKeys.id.value],
                    **result["customData"],
                }
                for result in results
            ]
            return pd.DataFrame(response_data)

    def get_molecules_from_id_or_name(
        self, molecule_set_id: str, return_as="dataframe"
    ) -> tuple[Union[pd.DataFrame, list], str]:
        molset_id = self.molecule_set_id_or_name(molecule_set_id, self.list_available())
        return self.get_molecules(molset_id, return_as), molset_id

    def add_molecules(
        self,
        molecule_set_id: id,
        data: MoleculeList,
    ) -> int:
        """Add additional molecules to the MoleculeSet.

        Parameters
        ----------
        molecule_set_id
            The unique id of the MoleculeSet.
        data
            MoleculeList giving Molecules to add.

        Returns
        -------
        Number of molecules not added due to exceeding the max number of
        molecules allowed in a MoleculeSet.

        """
        url = f"{self.molecule_set_url}/{molecule_set_id}/add_molecules/"
        response = self._session.post(
            url,
            json={
                "newMolecules": data,
            },
        )
        try:
            n_over_limit = response.json()["nOverLimit"]
        except Exception as e:
            raise ValueError(
                f"Add failed for molecule set {molecule_set_id}, with response: {response}"
            ) from e
        return n_over_limit

    def update_molecules(
        self, molecule_set_id: str, data: MoleculeUpdateList, overwrite=False
    ) -> list[str]:
        """Updates the custom data associated with the Molecules in a MoleculeSet.

        Parameters
        ----------
        molecule_set_id
            The unique id of the MoleculeSet.
        data
            MoleculeUpdateList giving MoleculeUpdates with update data for
            existing Molecules.
        overwrite
            If `True`, then customData will be entirely replaced with new data
            submitted; if `False`, then old and new customData will be combined,
            overwriting keys existing in both.

        Returns
        -------
        List of Molecule ids that were updated.

        """

        url = f"{self.molecule_set_url}/{molecule_set_id}/update_molecules/"

        response = self._session.patch(
            url, json={"moleculesToUpdate": data, "overwrite": overwrite}
        ).json()
        try:
            return response["moleculesUpdated"]

        except Exception as e:
            raise ValueError(
                f"Update failed for molecule set {molecule_set_id}, with response: {response}"
            ) from e

    def update_molecules_from_df_with_manifold_validation(
        self,
        molecule_set_id: str,
        df: pd.DataFrame,
        smiles_field: str = MoleculeSetKeys.smiles.value,
        id_field: str = MoleculeSetKeys.id.value,
        overwrite=False,
        bleached: bool = _POSTERA_COLUMN_BLEACHING_ACTIVE,  # NOTE: this is fix for postera bleaching see issues #629 #628
        debug_df_path: str = None,
    ) -> list[str]:
        if bleached:
            warnings.warn(
                "Fix currently applied for postera column name bleaching see issues #629 #628"
            )

        df = ManifoldAllowedTags.filter_dataframe_cols(
            df, allow=[smiles_field, id_field], bleached=bleached
        )

        if not ManifoldAllowedTags.all_in_values(
            df.columns, allow=[id_field, smiles_field], bleached=bleached
        ):
            raise ValueError(
                f"Columns in dataframe {df.columns} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedTags.get_values()}"
            )

        # fill nan values with empty string
        df = df.fillna("")

        # save debug df if requested
        if debug_df_path is not None:
            df.to_csv(debug_df_path, index=False)

        # make payload for postera
        mol_update_list = MoleculeUpdateList.from_pandas_df(
            df, smiles_field=smiles_field, id_field=id_field
        )

        # push updates to postera
        retcode = self.update_molecules(
            molecule_set_id, mol_update_list, overwrite=overwrite
        )

        if not retcode:
            raise ValueError(f"Update failed for molecule set {molecule_set_id}")

    def add_molecules_from_df_with_manifold_validation(
        self,
        molecule_set_id: str,
        df: pd.DataFrame,
        smiles_field: str = MoleculeSetKeys.smiles.value,
        id_field: str = MoleculeSetKeys.id.value,
        bleached: bool = _POSTERA_COLUMN_BLEACHING_ACTIVE,  # NOTE: this is fix for postera bleaching see issues #629 #628
        debug_df_path: str = None,
    ) -> int:
        if bleached:
            warnings.warn(
                "Fix currently applied for postera column name bleaching see issues #629 #628"
            )
        df = ManifoldAllowedTags.filter_dataframe_cols(
            df, allow=[smiles_field, id_field], bleached=bleached
        )

        if not ManifoldAllowedTags.all_in_values(
            df.columns, allow=[id_field, smiles_field], bleached=bleached
        ):
            raise ValueError(
                f"Columns in dataframe {df.columns} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedTags.get_values()}"
            )

        # fill nan values with empty string
        df = df.fillna("")

        # save debug df if requested
        if debug_df_path is not None:
            df.to_csv(debug_df_path, index=False)

        # make payload for postera
        mol_list = MoleculeList.from_pandas_df(
            df, smiles_field=smiles_field, id_field=id_field
        )

        # push updates to postera
        retcode = self.add_molecules(molecule_set_id, mol_list)

        if retcode:
            raise ValueError(f"Add failed for molecule set {molecule_set_id}")

    def create_molecule_set_from_df_with_manifold_validation(
        self,
        molecule_set_name: str,
        df: pd.DataFrame,
        smiles_field: str = MoleculeSetKeys.smiles.value,
        id_field: str = MoleculeSetKeys.id.value,
        bleached: bool = _POSTERA_COLUMN_BLEACHING_ACTIVE,  # NOTE: this is fix for postera bleaching see issues #629 #628
        debug_df_path: str = None,
    ) -> str:
        if bleached:
            warnings.warn(
                "Fix currently applied for postera column name bleaching see issues #629 #628"
            )
        df = ManifoldAllowedTags.filter_dataframe_cols(
            df, allow=[smiles_field, id_field], bleached=bleached
        )

        if not ManifoldAllowedTags.all_in_values(
            df.columns, allow=[id_field, smiles_field], bleached=bleached
        ):
            raise ValueError(
                f"Columns in dataframe {df.columns} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedTags.get_values()}"
            )

        # fill nan values with empty string
        df = df.fillna("")

        # save debug df if requested
        if debug_df_path is not None:
            df.to_csv(debug_df_path, index=False)

        # make payload for postera
        mol_list = MoleculeList.from_pandas_df(
            df, smiles_field=smiles_field, id_field=id_field
        )

        # push updates to postera
        id = self.create(molecule_set_name, mol_list, return_full=False)

        if not id:
            raise ValueError(f"Create failed for molecule set {molecule_set_name}")

        return id
