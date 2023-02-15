from .postera_api import PostEraAPI

from typing import Dict, Union, List
from typing_extensions import TypedDict
import json
import pandas as pd


class Molecule(TypedDict):
    """Data type to build MoleculeList"""

    smiles: str
    customData: Dict[str, Union[str, float, int]]


class MoleculeUpdate(TypedDict):
    """Data type to build MoleculeUpdateList"""

    id: str
    customData: Dict[str, Union[str, float, int]]


class MoleculeList(List[Molecule]):
    """Data type to pass to PostEra API in molecule set create"""

    def from_pandas_df(
        self,
        df: pd.DataFrame,
        smiles_field: str = None,
        first_entry: int = 0,
        last_entry: int = 0,
    ):

        #data = json.loads(df.to_json(orient="records"))

        self.extend(
            [
                {
                    "smiles": datum[smiles_field],
                    "customData": {
                        **{
                            key: value
                            for key, value in datum.items()
                            if key not in [smiles_field, "mol"]
                        },
                    },
                }
                for datum in df[first_entry:last_entry]
            ]
        )


class MoleculeUpdateList(List[MoleculeUpdate]):
    """Data type to pass to PostEra API in molecule set update_custom_data"""

    def from_pandas_df(
        self,
        df: pd.DataFrame,
        postera_id_field: str = None,
        first_entry: int = 0,
        last_entry: int = 0,
    ):

        #data = json.loads(df.to_json(orient="records"))

        self.extend(
            [
                {
                    "id": str(datum[postera_id_field]),
                    "customData": {
                        **{
                            key: value
                            for key, value in datum.items()
                            if key not in [postera_id_field, "mol"]
                        },
                    },
                }
                for datum in df[first_entry:last_entry]
            ]
        )


class MoleculeSet(PostEraAPI):
    """Connection and commands for PostEra Molecule Set API"""

    def __init__(self, *args, **kwargs):

        super(MoleculeSet, self).__init__(*args, **kwargs)

        self.molecule_set_url = f"{self.api_url}/moleculesets"

    def create(self, data: MoleculeList, set_name: str) -> str:

        create_url = f"{self.molecule_set_url}/"
        response = self._session.post(
            create_url,
            json={
                "molecules": data,
                "name": set_name,
            },
        ).json()

        return response["id"]

    def read_page(
            self, 
            molecule_set_id: str,
            page: int
        ) -> (pd.DataFrame, str):

        print(f"Getting page {page}")
        read_url = f"{self.molecule_set_url}/{molecule_set_id}/get_all_molecules/"
        response = self._session.get(read_url, params={"page": page}).json()

        return response["results"], response["paginationInfo"]["hasNext"]

    def read(
            self,
            molecule_set_id: str
        ) -> pd.DataFrame:

        page = 0
        has_next = True
        all_results = []

        while has_next:
            page += 1
            result, has_next = self.read_page(molecule_set_id, page)
            all_results.extend(result)

        response_data = [
            {
                "SMILES": result["smiles"],
                "postera_molecule_id": result["id"],
                **result["customData"],
            }
            for result in all_results
        ]

        result_df = pd.DataFrame(response_data)

        return result_df

    def update_custom_data(
        self, 
        molecule_set_id: str, 
        data: MoleculeUpdateList, 
        overwrite=False
    ) -> List[float]:
        """Updates the custom data associated with the Molecules in a MoleculeSet.

        Parameters
        ----------
        molecule_set_id
            The unique id of the MoleculeSet

        """

        update_url = f"{self.molecule_set_url}/{molecule_set_id}/update_molecules/"
        response = self._session.patch(
            update_url, json={"moleculesToUpdate": data, "overwrite": overwrite}
        ).json()

        return response["moleculesUpdated"]
