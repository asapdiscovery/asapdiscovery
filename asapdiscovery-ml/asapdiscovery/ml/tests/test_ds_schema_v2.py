import pytest

from asapdiscovery.data.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
)
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.schema_v2.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    DatasetType,
)


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


def test_docked_dataset_config(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test1"},
        ligand_kwargs={"compound_name": "test1"},
    )
    c2 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test2"},
        ligand_kwargs={"compound_name": "test2"},
    )

    config = DatasetConfig(ds_type=DatasetType.structural, input_data=[c1, c2])
    dd = config.build()

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test1"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test2"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])


def test_docked_dataset_config_exp_dict(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test1"},
        ligand_kwargs={"compound_name": "test1"},
    )
    c2 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test2"},
        ligand_kwargs={"compound_name": "test2"},
    )

    exp_data = ExperimentalCompoundDataUpdate(
        compounds=[
            ExperimentalCompoundData(
                compound_id="test1", experimental_data={"pIC50": 5}
            ),
            ExperimentalCompoundData(
                compound_id="test2", experimental_data={"pIC50": 6}
            ),
        ]
    )

    config = DatasetConfig(
        ds_type=DatasetType.structural, input_data=[c1, c2], exp_data=exp_data
    )
    dd = config.build()

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test1"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pIC50"] == 5

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test2"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pIC50"] == 6
