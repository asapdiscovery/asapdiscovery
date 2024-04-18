import uuid
from unittest.mock import patch

import pandas as pd
import pytest
from asapdiscovery.data.services.postera.molecule_set import (
    Molecule,
    MoleculeList,
    MoleculeSetAPI,
    MoleculeUpdateList,
)
from asapdiscovery.data.services.services_config import PosteraSettings
from requests import Session


class TestMoleculeList:
    def test_from_pandas_df(self):
        expected_molecule_list = [
            {
                "smiles": "SMILES1",
                "customData": {
                    "custom_fieldA": "CUSTOMA_1",
                    "custom_fieldB": 1.0,
                    "custom_filedC": 3,
                },
            },
            {
                "smiles": "SMILES2",
                "customData": {
                    "custom_fieldA": "CUSTOMA_2",
                    "custom_fieldB": 2.0,
                    "custom_filedC": 4,
                },
            },
        ]

        test_data = {
            "smiles_field": ["SMILES1", "SMILES2"],
            "custom_fieldA": ["CUSTOMA_1", "CUSTOMA_2"],
            "custom_fieldB": [1.0, 2.0],
            "custom_filedC": [3, 4],
        }

        df = pd.DataFrame.from_dict(test_data)

        molecule_list = MoleculeList.from_pandas_df(df, smiles_field="smiles_field")

        assert molecule_list == expected_molecule_list


class TestMoleculeUpdateList:
    def test_from_pandas_df(self):
        expected_molecule_update_list = [
            {
                "id": "1",
                "customData": {
                    "custom_fieldA": "CUSTOMA_1",
                    "custom_fieldB": 1.0,
                    "custom_filedC": 3,
                },
            },
            {
                "id": "2",
                "customData": {
                    "custom_fieldA": "CUSTOMA_2",
                    "custom_fieldB": 2.0,
                    "custom_filedC": 4,
                },
            },
        ]

        test_data = {
            "id_field": ["1", "2"],
            "custom_fieldA": ["CUSTOMA_1", "CUSTOMA_2"],
            "custom_fieldB": [1.0, 2.0],
            "custom_filedC": [3, 4],
        }

        df = pd.DataFrame.from_dict(test_data)
        molecule_update_list = MoleculeUpdateList.from_pandas_df(
            df, id_field="id_field"
        )

        assert molecule_update_list == expected_molecule_update_list


class TestMoleculeSet:
    @pytest.fixture
    def moleculesetapi(self):
        return MoleculeSetAPI("mock_api_url", "mock_api_version", "mock_api_key")

    def test_from_settings(self):
        postera_settings = PosteraSettings(
            POSTERA_API_KEY="mock_api_key",
            POSTERA_API_URL="mock_api_url",
            POSTERA_API_VERSION="mock_api_version",
        )
        _ = MoleculeSetAPI.from_settings(postera_settings)

    @patch.object(Session, "post")
    def test_create(self, mock_post, moleculesetapi):
        # create a MoleculeList for submission
        moleculeset_id = str(uuid.uuid4())

        mock_post.return_value.json.return_value = {"id": moleculeset_id}

        mols = MoleculeList([Molecule(smiles=i * "C") for i in range(1, 4)])

        molset_id = moleculesetapi.create("molset_name", mols)

        assert molset_id == moleculeset_id

    @patch.object(Session, "get")
    def test_list(self, mock_get, moleculesetapi):
        mock_get.return_value.json.return_value = {
            "results": [
                {
                    "id": "497f6eca-6276-4993-bfeb-53cbbbba6f08",
                    "created": "2019-08-24T14:15:22Z",
                    "updated": "2019-08-24T14:15:22Z",
                    "link": "string",
                    "name": "test_set",
                    "molecules": [
                        {
                            "smiles": "string",
                            "customData": {
                                "property1": "string",
                                "property2": "string",
                            },
                        }
                    ],
                }
            ],
            "paginationInfo": {
                "page": 0,
                "numberOfPages": 0,
                "pageNumbersList": [0],
                "count": 0,
                "hasNext": False,
            },
        }

        output = moleculesetapi.list_available()

        assert output == {"497f6eca-6276-4993-bfeb-53cbbbba6f08": "test_set"}

    @patch.object(Session, "get")
    def test_get(self, mock_get, moleculesetapi):
        metadata = {
            "id": "497f6eca-6276-4993-bfeb-53cbbbba6f08",
            "created": "2019-08-24T14:15:22Z",
            "updated": "2019-08-24T14:15:22Z",
            "link": "string",
            "name": "test_set",
        }
        mock_get.return_value.json.return_value = metadata

        output = moleculesetapi.get("497f6eca-6276-4993-bfeb-53cbbbba6f08")
        assert output == metadata

    @patch.object(Session, "get")
    def test_get_molecules(self, mock_get, moleculesetapi):
        mock_get.return_value.json.side_effect = [
            {
                # page with 2 molecules
                "results": [
                    {
                        "smiles": "SMILES1",
                        "id": "ID1",
                        "customData": {"field": "DATA1"},
                    },
                    {
                        "smiles": "SMILES2",
                        "id": "ID2",
                        "customData": {"field": "DATA2"},
                    },
                ],
                "paginationInfo": {"hasNext": True},
            },
            {
                # page with 1 molecule
                "results": [
                    {"smiles": "SMILES3", "id": "ID3", "customData": {"field": "DATA3"}}
                ],
                "paginationInfo": {"hasNext": False},
            },
        ]

        expected_dict = {
            "smiles": ["SMILES1", "SMILES2", "SMILES3"],
            "id": ["ID1", "ID2", "ID3"],
            "field": ["DATA1", "DATA2", "DATA3"],
        }

        expected_output_df = pd.DataFrame.from_dict(expected_dict)
        output_df = moleculesetapi.get_molecules("mock_molecule_set_id")

        pd.testing.assert_frame_equal(expected_output_df, output_df)
