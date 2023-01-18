import unittest
from unittest.mock import patch

from requests import Session

import pandas as pd

from asapdiscovery.postera_api.src.molecule_set_crud import MoleculeList, MoleculeUpdateList, MoleculeSetCRUD

class TestMoleculeList(unittest.TestCase):

    def test_from_pandas_df(self):

        expected_molecule_list = [
            {
                'smiles': 'SMILES1',
                'customData': {
                    'custom_fieldA': 'CUSTOMA_1',
                    'custom_fieldB': 1.0,
                    'custom_filedC': 3
                }
            },

            {
                'smiles': 'SMILES2',
                'customData': {
                    'custom_fieldA': 'CUSTOMA_2',
                    'custom_fieldB': 2.0,
                    'custom_filedC': 4
                }
            }
        ]

        test_data = {
            "smiles_field": ["SMILES1", "SMILES2"],
            "custom_fieldA": ["CUSTOMA_1", "CUSTOMA_2"],
            "custom_fieldB": [1.0, 2.0],
            "custom_filedC": [3, 4]
        }

        df = pd.DataFrame.from_dict(test_data)
        molecule_list = MoleculeList()
        molecule_list.from_pandas_df(df, smiles_field="smiles_field", first_entry=0, last_entry=len(df))

        self.assertListEqual(molecule_list, expected_molecule_list,
                             "MoleculeList.from_pandas_df produces incorrect list")


class TestMoleculeUpdateList(unittest.TestCase):

    def test_from_pandas_df(self):

        expected_molecule_update_list = [
            {
                'id': '1',
                'customData': {
                    'custom_fieldA': 'CUSTOMA_1',
                    'custom_fieldB': 1.0,
                    'custom_filedC': 3
                }
            },
            {
                'id': '2',
                'customData':{
                    'custom_fieldA': 'CUSTOMA_2',
                    'custom_fieldB': 2.0,
                    'custom_filedC': 4
                }
            }
        ]

        test_data = {
            "id_field": ["1", "2"],
            "custom_fieldA": ["CUSTOMA_1", "CUSTOMA_2"],
            "custom_fieldB": [1.0, 2.0],
            "custom_filedC": [3, 4]
        }

        df = pd.DataFrame.from_dict(test_data)
        molecule_update_list = MoleculeUpdateList()
        molecule_update_list.from_pandas_df(df, postera_id_field="id_field", first_entry=0, last_entry=len(df))

        self.assertListEqual(molecule_update_list, expected_molecule_update_list,
                             "MoleculeUpdateList.from_pandas_df produces incorrect list")


class TestMoleculeSetCRUD(unittest.TestCase):

    @patch.object(Session, 'get')
    def test_read(self, mock_get):

        mock_get.return_value.json.side_effect = [
            {
                # page with 2 molecules
                "results": [
                    {
                        "smiles": "SMILES1",
                        "id": "ID1",
                        "customData": {"field": "DATA1"}
                    },
                    {
                        "smiles": "SMILES2",
                        "id": "ID2",
                        "customData": {"field": "DATA2"}
                    }
                ],
                "paginationInfo": {"hasNext" : True}
            },
            {
                # page with 1 molecule
                "results": [
                    {
                        "smiles": "SMILES3",
                        "id": "ID3",
                        "customData": {"field": "DATA3"}
                    }
                ],
                "paginationInfo": {"hasNext": False}
            }
        ]

        expected_dict = {
            "SMILES": ["SMILES1", "SMILES2", "SMILES3"],
            "postera_molecule_id": ["ID1", "ID2", "ID3"],
            "field": ["DATA1", "DATA2", "DATA3"]
        }

        expected_output_df = pd.DataFrame.from_dict(expected_dict)

        molecule_set = MoleculeSetCRUD("mock_api_url", "mock_api_version", "mock_api_key")
        output_df = molecule_set.read("mock_molecule_set_id")

        pd.testing.assert_frame_equal(expected_output_df, output_df)