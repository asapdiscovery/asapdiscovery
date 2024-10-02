import logging
import warnings
from typing import Dict, Optional, Tuple, Union  # noqa: F401

import pandas as pd
from asapdiscovery.data.services.web_utils import _BaseWebAPI
from asapdiscovery.data.util.stringenum import StringEnum
from typing_extensions import TypedDict

from .manifold_data_validation import ManifoldAllowedTags

logger = logging.getLogger(__name__)


def _batch(iterable, n=1):
    l = len(iterable)  # noqa: E741
    for ndx in range(0, l, n):  # noqa: E741
        yield iterable[ndx : min(ndx + n, l)]


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


class MoleculeSetAPI(_BaseWebAPI):
    """Connection and commands for PostEra Molecule Set API"""

    @staticmethod
    def _check_response_for_perm_error(response: dict):
        detail = response.get("detail")
        if detail and "You do not have permission" in detail:
            raise ValueError(
                f"User does not have permission to perform this operation in the PostEra API, check API key and user permissions. Response: {response}"
            )

    @classmethod
    def token_name(cls) -> str:
        return "X-API-KEY"

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
            timeout=self.timeout,
        )
        response_json = response.json()
        logger.debug(
            f"Postera MoleculeSetAPI.create response: {response_json}, status code: {response.status_code}"
        )
        self._check_response_for_perm_error(response_json)
        response.raise_for_status()

        if return_full:
            return response_json
        else:
            return response_json[MoleculeSetKeys.id.value]

    def _read_page(self, url: str, page: int) -> tuple[pd.DataFrame, str]:
        response = self._session.get(url, params={"page": page}, timeout=self.timeout)
        response.raise_for_status()
        response_json = response.json()
        return response_json["results"], response_json["paginationInfo"]["hasNext"]

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
            timeout=self.timeout,
        )
        response_json = response.json()
        logger.debug(
            f"Postera MoleculeSetAPI.get response: {response_json}, status code: {response.status_code}"
        )
        self._check_response_for_perm_error(response_json)
        response.raise_for_status()

        return response_json

    def destroy(self, molecule_set_id: str) -> None:
        """Delete a MoleculeSet.

        Parameters
        ----------
        molecule_set_id
            The unique id of the MoleculeSet

        """
        url = f"{self.molecule_set_url}/{molecule_set_id}"
        response = self._session.delete(url, timeout=self.timeout)
        # no response body for delete
        logger.debug(
            f"Postera MoleculeSetAPI.destroy response: {response}, status code: {response.status_code}"
        )
        response.raise_for_status()

    def get_molecules(
        self, molecule_set_id: str, return_as="dataframe"
    ) -> Union[pd.DataFrame, list]:
        """Pull the full contents of a MoleculeSet as a DataFrame.

        Parameters
        ----------
        molecule_set_id
            The unique id of the MoleculeSet
        return_as : {'dataframe', 'list'}
            Whether to return the molecules as a DataFrame or a list.

        """

        if return_as not in ("dataframe", "list"):
            raise ValueError("`return_as` must be either 'dataframe' or 'list'")

        url = f"{self.molecule_set_url}/{molecule_set_id}/get_all_molecules/"

        results = self._collate(url)

        if return_as == "list":
            return results
        elif return_as == "dataframe":

            response_data = []
            for result in results:
                data = {
                    MoleculeSetKeys.smiles.value: result[MoleculeSetKeys.smiles.value],
                    MoleculeSetKeys.id.value: result[MoleculeSetKeys.id.value],
                }
                # rare case where customData has the same key name as a reserved key like id or smiles
                for key, value in result["customData"].items():
                    if key in MoleculeSetKeys.get_values():
                        warnings.warn(
                            f"Custom data key name {key} is the same as a reserved key name, skipping.."
                        )
                    else:
                        data[key] = value

                response_data.append(data)

            return pd.DataFrame(response_data)

    def get_id_from_name(self, name: str) -> str:
        """Get the unique id of a MoleculeSet from its human-readable name.

        Parameters
        ----------
        name
            The human-readable name of the MoleculeSet.

        Returns
        -------
        str
            The unique id of the MoleculeSet.

        """
        avail = self.list_available(return_full=False)
        avail_rev = {v: k for k, v in avail.items()}
        id = avail_rev.get(name)
        if id is None:
            raise ValueError(f"Molecule set with name {name} not found in PostEra")
        return id

    def get_name_from_id(self, id: str) -> str:
        """Get the human-readable name of a MoleculeSet from its unique id.

        Parameters
        ----------
        id
            The unique id of the MoleculeSet.

        Returns
        -------
        str
            The human-readable name of the MoleculeSet.

        """
        avail = self.list_available(return_full=False)
        name = avail.get(id)
        if name is None:
            raise ValueError(f"Molecule set with id {id} not found in PostEra")
        return name

    def get_molecules_from_id_or_name(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        return_as="dataframe",
    ) -> tuple[Union[pd.DataFrame, list], str]:
        """
        Get the molecules from a molecule set by either id or name.

        Parameters
        ----------
        id
            The unique id of the MoleculeSet.
        name
            The human-readable name of the MoleculeSet.
        return_as : {'dataframe', 'list'}
            Whether to return the molecules as a DataFrame or a list.

        Returns
        -------
        Union[pd.DataFrame, list]
            The molecules in the molecule set.
        """
        if id is None and name is None:
            raise ValueError("Either id or name must be set")

        if id is not None and name is not None:
            raise ValueError("Only one of id or name can be set")

        if name is not None:
            molset_id = self.get_id_from_name(name)
            if molset_id is None:
                raise ValueError(f"Molecule set with name {name} not found in PostEra")

        if id is not None:
            molset_id = id

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
            timeout=self.timeout,
        )
        response_json = response.json()
        logger.debug(
            f"Postera MoleculeSetAPI.add_molecules response: {response_json}, status code: {response.status_code}"
        )
        self._check_response_for_perm_error(response_json)

        try:
            n_over_limit = response_json["nOverLimit"]
        except Exception as e:
            raise ValueError(
                f"Add failed for molecule set {molecule_set_id}, with response: {response}"
            ) from e
        response.raise_for_status()
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

        molecules_updated = []
        for data_batch in _batch(data, n=100):
            response = self._session.patch(
                url,
                json={"moleculesToUpdate": data_batch, "overwrite": overwrite},
                timeout=self.timeout,
            )
            response_json = response.json()

            logger.debug(
                f"Postera MoleculeSetAPI.update_molecules response: {response_json}, status code: {response.status_code}"
            )
            self._check_response_for_perm_error(response_json)
            response.raise_for_status()

            try:
                updated = response_json["moleculesUpdated"]
                molecules_updated.extend(updated)

            except Exception as e:
                raise ValueError(
                    f"Update failed for molecule set batch {molecule_set_id}, with response: {response_json}, status code: {response.status_code}"
                ) from e

        return molecules_updated

    def update_molecules_from_df_with_manifold_validation(
        self,
        molecule_set_id: str,
        df: pd.DataFrame,
        smiles_field: str = MoleculeSetKeys.smiles.value,
        id_field: str = MoleculeSetKeys.id.value,
        overwrite=False,
        debug_df_path: str = None,
    ) -> list[str]:
        df = ManifoldAllowedTags.filter_dataframe_cols(
            df, allow=[smiles_field, id_field]
        )

        if not ManifoldAllowedTags.all_in_values(
            df.columns, allow=[id_field, smiles_field]
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
        debug_df_path: str = None,
    ) -> int:
        df = ManifoldAllowedTags.filter_dataframe_cols(
            df, allow=[smiles_field, id_field]
        )

        if not ManifoldAllowedTags.all_in_values(
            df.columns, allow=[id_field, smiles_field]
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
        debug_df_path: str = None,
    ) -> str:
        df = ManifoldAllowedTags.filter_dataframe_cols(
            df, allow=[smiles_field, id_field]
        )

        if not ManifoldAllowedTags.all_in_values(
            df.columns, allow=[id_field, smiles_field]
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
