import pytest
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    DatasetSplitterType,
    DatasetType,
)


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


@pytest.fixture(scope="session")
def ligand_sdf():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    return sdf


def test_manual_split_docked_dataset(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test1"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test2"},
        ligand_kwargs={"compound_name": "test"},
    )
    c3 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test3"},
        ligand_kwargs={"compound_name": "test"},
    )

    ds = DatasetConfig(
        ds_type=DatasetType.structural, input_data=[c1, c2, c3], grouped=False
    ).build()

    ds_splitter = DatasetSplitterConfig(
        split_type="manual",
        split_dict={
            "train": [("test1", "test")],
            "val": [("test2", "test")],
            "test": [("test3", "test")],
        },
    )

    ds_train, ds_val, ds_test = ds_splitter.split(ds)

    assert len(ds_train) == 1
    assert len(ds_val) == 1
    assert len(ds_test) == 1
