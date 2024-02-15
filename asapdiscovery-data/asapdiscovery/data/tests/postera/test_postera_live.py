import pytest
import os
import pandas as pd
import numpy as np
from uuid import uuid4
import random
from asapdiscovery.data.postera.molecule_set import (
    MoleculeSetAPI,
    MoleculeList,
    MoleculeUpdateList,
)
from asapdiscovery.data.postera.manifold_data_validation import ManifoldAllowedTags
from asapdiscovery.data.services_config import PosteraSettings

from hashlib import sha256

# WARNING IMPORTANT: - this is a live test and will make real requests to the POSTERA API
# A sanboxed API key is required to run this test, DO NOT USE A PRODUCTION API KEY
# we have a second environment variable to stop you accidentally running this test with a production API key
# DO NOT REMOVE


@pytest.mark.skipif(not os.getenv("POSTERA_API_KEY"), reason="No POSTERA_API_KEY")
@pytest.mark.skipif(
    not os.getenv("POSTERA_API_KEY_IS_SANDBOX"), reason="No POSTERA_API_KEY_IS_SANDBOX"
)
@pytest.mark.skipif(
    os.getenv("POSTERA_API_KEY_HASH")
    != sha256(os.getenv("POSTERA_API_KEY").encode()).hexdigest(),
    reason="POSTERA_API_KEY is not the sandbox key",
)
class TestPosteraLive:

    @pytest.fixture()
    def postera_settings(self):
        return PosteraSettings()

    @pytest.fixture()
    def simple_moleculeset_data(self):
        return pd.DataFrame(
            {
                "smiles_field": ["CCCC", "CCCCCCCC"],
                "id_field": ["1", "2"],
                "custom_fieldA": ["CUSTOMA_1", "CUSTOMA_2"],
                "custom_fieldB": [1.0, 2.0],
                "custom_fieldC": [3, 4],
                "custom_fieldD": [True, False],
            }
        )

    @pytest.fixture()
    def simple_moleculeset_molecule_list(self, simple_moleculeset_data):
        return MoleculeList.from_pandas_df(
            simple_moleculeset_data, smiles_field="smiles_field", id_field="id_field"
        )

    @pytest.fixture()
    def live_postera_ms_api_instance(self, postera_settings):
        ms_api = MoleculeSetAPI.from_settings(postera_settings)
        return ms_api

    def test_simple_moleculeset_create(
        self, live_postera_ms_api_instance, simple_moleculeset_molecule_list
    ):
        # make a random  string for the molecule set name
        molecule_set_name = str(uuid4())
        ms_api = live_postera_ms_api_instance
        uuid = ms_api.create(molecule_set_name, simple_moleculeset_molecule_list)
        assert uuid is not None
        assert ms_api.exists(molecule_set_name, by="name")
        assert ms_api.exists(uuid, by="id")
        # clean up
        ms_api.destroy(uuid)

    @pytest.fixture()
    def simple_moleculeset(
        self, live_postera_ms_api_instance, simple_moleculeset_molecule_list
    ):
        # fixture of the above test
        # make a random  string for the molecule set name
        molecule_set_name = str(uuid4())
        ms_api = live_postera_ms_api_instance
        uuid = ms_api.create(molecule_set_name, simple_moleculeset_molecule_list)
        yield molecule_set_name, uuid
        # clean up
        ms_api.destroy(uuid)

    def test_exists(self, live_postera_ms_api_instance, simple_moleculeset):
        molecule_set_name, uuid = simple_moleculeset
        ms_api = live_postera_ms_api_instance
        assert ms_api.exists(molecule_set_name, by="name")
        assert ms_api.exists(uuid, by="id")

    def test_garbage_api_key(self):
        postera_settings_garbage = PosteraSettings(POSTERA_API_KEY="garbage_api")
        ms_api = MoleculeSetAPI.from_settings(postera_settings_garbage)
        assert ms_api.list_available() == {}

    def test_get(self, live_postera_ms_api_instance, simple_moleculeset):
        molecule_set_name, uuid = simple_moleculeset
        ms_api = live_postera_ms_api_instance
        molecules = ms_api.get(uuid)
        assert molecules["id"] == uuid
        assert molecules["name"] == molecule_set_name

    def test_get_molecules(self, live_postera_ms_api_instance, simple_moleculeset):
        molecule_set_name, uuid = simple_moleculeset
        ms_api = live_postera_ms_api_instance
        molecules = ms_api.get_molecules(uuid, return_as="dataframe")
        assert molecules["custom_fieldA"].tolist() == ["CUSTOMA_1", "CUSTOMA_2"]
        assert molecules["custom_fieldB"].tolist() == [1.0, 2.0]
        assert molecules["custom_fieldC"].tolist() == [3, 4]
        # bool is cast to string in the API
        assert molecules["custom_fieldD"].tolist() == ["True", "False"]

    def test_update_set_equal(self, live_postera_ms_api_instance, simple_moleculeset):
        # update with no new molecules
        _, uuid = simple_moleculeset
        ms_api = live_postera_ms_api_instance
        # id and smiles fields will be set to their postera values on pulldown
        molecules_df = ms_api.get_molecules(uuid, return_as="dataframe")
        molecules_df["custom_fieldA"] = ["NEW_CUSTOMA1", "NEW_CUSTOMA2"]
        mlu = MoleculeUpdateList.from_pandas_df(
            molecules_df, id_field="id", smiles_field="smiles"
        )
        ms_api.update_molecules(uuid, mlu)
        updated_molecules = ms_api.get_molecules(uuid, return_as="dataframe")
        assert updated_molecules["custom_fieldA"].tolist() == [
            "NEW_CUSTOMA1",
            "NEW_CUSTOMA2",
        ]

    def test_update_set_strict_subset(
        self, live_postera_ms_api_instance, simple_moleculeset
    ):
        # update one subset of molecules
        molecule_set_name, uuid = simple_moleculeset
        ms_api = live_postera_ms_api_instance
        # id and smiles fields will be set to their postera values on pulldown
        molecules_df = ms_api.get_molecules(uuid, return_as="dataframe")
        # grab the first row
        new_molecules_df = molecules_df.iloc[:1]
        new_molecules_df["custom_fieldA"] = ["NEW_CUSTOMA3"]
        mlu = MoleculeUpdateList.from_pandas_df(new_molecules_df)
        ms_api.update_molecules(uuid, mlu)
        updated_molecules = ms_api.get_molecules(uuid, return_as="dataframe")
        assert any(updated_molecules["custom_fieldA"] != molecules_df["custom_fieldA"])
        assert updated_molecules["custom_fieldA"].tolist() == [
            "NEW_CUSTOMA3",
            "CUSTOMA_2",
        ]

    def test_update_set_strict_subset_overwrite(
        self, live_postera_ms_api_instance, simple_moleculeset
    ):
        # update one subset of molecules
        molecule_set_name, uuid = simple_moleculeset
        ms_api = live_postera_ms_api_instance
        # id and smiles fields will be set to their postera values on pulldown
        molecules_df = ms_api.get_molecules(uuid, return_as="dataframe")
        # grab the first row
        new_molecules_df = molecules_df.iloc[:1]
        columns_to_keep = ["id", "smiles", "custom_fieldA"]
        new_molecules_df = new_molecules_df[columns_to_keep]
        new_molecules_df["custom_fieldA"] = ["NEW_CUSTOMA3"]
        mlu = MoleculeUpdateList.from_pandas_df(new_molecules_df)
        # remove all other custom data other than what we are writing
        ms_api.update_molecules(uuid, mlu, overwrite=True)
        updated_molecules = ms_api.get_molecules(uuid, return_as="dataframe")
        assert updated_molecules["custom_fieldA"].tolist() == [
            "NEW_CUSTOMA3",
            "CUSTOMA_2",
        ]
        # check that the other custom fields are gone on molecules that were not updated
        assert np.array_equal(
            updated_molecules["custom_fieldB"].tolist(), [np.nan, 2.0], equal_nan=True
        )

    def test_add(self, live_postera_ms_api_instance, simple_moleculeset):
        # add a molecule to the set
        molecule_set_name, uuid = simple_moleculeset
        ms_api = live_postera_ms_api_instance

        # make sopme new molecules
        new_molecules = pd.DataFrame(
            {
                "smiles_field": ["CCCCCF"],
                "id_field": ["3"],
                "custom_fieldA": ["NEW_CUSTOMA3"],
                "custom_fieldB": [3.0],
                "custom_fieldC": [5],
                "custom_fieldD": [False],
            }
        )

        # make payload for postera
        mol_list = MoleculeList.from_pandas_df(
            new_molecules, smiles_field="smiles_field", id_field="id_field"
        )

        # push updates to postera
        ms_api.add_molecules(uuid, mol_list)
        ret_df = ms_api.get_molecules(uuid, return_as="dataframe")
        # check the smiles is there
        assert "CCCCCF" in ret_df["smiles"].tolist()

    def test_create_with_data_validation(self, live_postera_ms_api_instance):
        # add a molecule to the set
        ms_api = live_postera_ms_api_instance

        # make random data for each element
        fields = ManifoldAllowedTags.get_values()
        data = {field: random.randint(0, 1) for field in fields}
        # turn it into a dataframe
        new_molecules = pd.DataFrame(
            {"smiles_field": ["CCCCCF"], "id_field": ["3"], **data}
        )

        uuid = ms_api.create_molecule_set_from_df_with_manifold_validation(
            molecule_set_name=str(uuid4()),
            df=new_molecules,
            id_field="id_field",
            smiles_field="smiles_field",
        )
        ret_df = ms_api.get_molecules(uuid, return_as="dataframe")
        # check all the data is there
        assert ret_df.sort_index(inplace=True) == new_molecules.sort_index(inplace=True)
        ms_api.destroy(uuid)

    def test_create_with_data_validation_drops_extra(
        self, live_postera_ms_api_instance
    ):
        # add a molecule to the set
        ms_api = live_postera_ms_api_instance

        # make random data for each element
        fields = ManifoldAllowedTags.get_values()
        data = {field: random.randint(0, 1) for field in fields}
        # turn it into a dataframe
        new_molecules = pd.DataFrame(
            {
                "smiles_field": ["CCCCCF"],
                "id_field": ["3"],
                "extra_field": ["extra"],
                **data,
            }
        )

        uuid = ms_api.create_molecule_set_from_df_with_manifold_validation(
            molecule_set_name=str(uuid4()),
            df=new_molecules,
            id_field="id_field",
            smiles_field="smiles_field",
        )
        ret_df = ms_api.get_molecules(uuid, return_as="dataframe")
        ms_api.destroy(uuid)
        # check all the data is there
        assert "extra_field" not in ret_df.columns

    def test_add_with_data_validation_drops_extra(self, live_postera_ms_api_instance):
        # add a molecule to the set
        ms_api = live_postera_ms_api_instance

        # make random data for each element
        fields = ManifoldAllowedTags.get_values()
        data = {field: random.randint(0, 1) for field in fields}
        # turn it into a dataframe
        new_molecules = pd.DataFrame({"smiles": ["CCCCCF"], "id": ["3"], **data})

        uuid = ms_api.create_molecule_set_from_df_with_manifold_validation(
            molecule_set_name=str(uuid4()),
            df=new_molecules,
            id_field="id",
            smiles_field="smiles",
        )

        # add another molecule
        new_molecules_to_add = pd.DataFrame(
            {"smiles": ["CCCCCCCCF"], "id": ["4"], "extra_field": ["extra"], **data}
        )

        ms_api.add_molecules_from_df_with_manifold_validation(
            molecule_set_id=uuid,
            df=new_molecules_to_add,
            id_field="id",
            smiles_field="smiles",
        )

        ret_df = ms_api.get_molecules(uuid, return_as="dataframe")
        ms_api.destroy(uuid)
        # check all the data is there
        assert "extra_field" not in ret_df.columns
        print(ret_df)
        print(ret_df.columns)
        # check both smiles are there
        assert "CCCCCF" in ret_df["smiles"].tolist()
        assert "CCCCCCCCF" in ret_df["smiles"].tolist()

    def test_update_with_data_validation(self, live_postera_ms_api_instance):
        # add a molecule to the set
        ms_api = live_postera_ms_api_instance

        # make random data for each element
        fields = ManifoldAllowedTags.get_values()
        data = {field: random.randint(0, 1) for field in fields}
        # turn it into a dataframe
        new_molecules = pd.DataFrame({"smiles": ["CCCCCF"], "id": ["3"], **data})

        uuid = ms_api.create_molecule_set_from_df_with_manifold_validation(
            molecule_set_name=str(uuid4()),
            df=new_molecules,
            id_field="id",
            smiles_field="smiles",
        )

        ret_df = ms_api.get_molecules(uuid, return_as="dataframe")

        # update the fields
        updated_data = {field: random.randint(0, 1) for field in fields}

        # grab the id field from ret_df
        mol_uuid = ret_df["id"].iloc[0]

        updated_molecules = pd.DataFrame(
            {"smiles": ["CCCCCF"], "id": [mol_uuid], **updated_data}
        )

        ms_api.update_molecules_from_df_with_manifold_validation(
            molecule_set_id=uuid,
            df=updated_molecules,
            id_field="id",
            smiles_field="smiles",
        )

        # get the updated data
        ret_df_updated = ms_api.get_molecules(uuid, return_as="dataframe")
        ms_api.destroy(uuid)
        ret_df = ret_df.sort_index()
        ret_df_updated = ret_df_updated.sort_index(inplace=True)
        assert any(ret_df == ret_df_updated)
