import pytest
from asapdiscovery.data.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
)
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset, GroupedDockedDataset
from asapdiscovery.ml.schema_v2.config import (
    DatasetConfig,
    DatasetSplitterConfig,
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
    assert isinstance(dd, DockedDataset)

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test1"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test2"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[1] > 0


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
    assert isinstance(dd, DockedDataset)

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test1"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0
    assert pose["pIC50"] == 5

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test2"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[1] > 0
    assert pose["pIC50"] == 6


def test_grouped_docked_dataset_config(complex_pdb):
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

    config = DatasetConfig(
        ds_type=DatasetType.structural, input_data=[c1, c2], grouped=True
    )
    dd = config.build()
    assert isinstance(dd, GroupedDockedDataset)

    assert len(dd) == 1

    compound_id, pose_list = next(iter(dd))
    assert compound_id == "test"
    assert len(pose_list) == 2

    assert pose_list[0]["compound"] == ("test1", "test")
    assert (
        pose_list[0]["pos"].shape[0]
        == len(pose_list[0]["z"])
        == len(pose_list[0]["lig"])
    )
    assert pose_list[0]["pos"].shape[0] > 0

    assert pose_list[1]["compound"] == ("test2", "test")
    assert (
        pose_list[1]["pos"].shape[0]
        == len(pose_list[1]["z"])
        == len(pose_list[1]["lig"])
    )
    assert pose_list[1]["pos"].shape[0] > 0


def test_grouped_docked_dataset_config_exp_dict(complex_pdb):
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

    exp_data = ExperimentalCompoundDataUpdate(
        compounds=[
            ExperimentalCompoundData(
                compound_id="test", experimental_data={"pIC50": 5}
            ),
        ]
    )

    config = DatasetConfig(
        ds_type=DatasetType.structural,
        input_data=[c1, c2],
        exp_data=exp_data,
        grouped=True,
    )
    dd = config.build()
    assert isinstance(dd, GroupedDockedDataset)

    assert len(dd) == 1

    compound_id, pose_list = next(iter(dd))
    assert compound_id == "test"
    assert len(pose_list) == 2

    assert pose_list[0]["compound"] == ("test1", "test")
    assert (
        pose_list[0]["pos"].shape[0]
        == len(pose_list[0]["z"])
        == len(pose_list[0]["lig"])
    )
    assert pose_list[0]["pos"].shape[0] > 0
    assert pose_list[0]["pIC50"] == 5

    assert pose_list[1]["compound"] == ("test2", "test")
    assert (
        pose_list[1]["pos"].shape[0]
        == len(pose_list[1]["z"])
        == len(pose_list[1]["lig"])
    )
    assert pose_list[1]["pos"].shape[0] > 0
    assert pose_list[1]["pIC50"] == 5


def test_graph_dataset_config(ligand_sdf):
    lig1 = Ligand.from_sdf(ligand_sdf, compound_name="test1")
    lig2 = Ligand.from_sdf(ligand_sdf, compound_name="test2")

    config = DatasetConfig(ds_type=DatasetType.graph, input_data=[lig1, lig2])
    dd = config.build()
    assert isinstance(dd, GraphDataset)

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test1"

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test2"


def test_graph_dataset_config_exp_dict(ligand_sdf):
    lig1 = Ligand.from_sdf(ligand_sdf, compound_name="test1")
    lig2 = Ligand.from_sdf(ligand_sdf, compound_name="test2")

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
        ds_type=DatasetType.graph, input_data=[lig1, lig2], exp_data=exp_data
    )
    dd = config.build()
    assert isinstance(dd, GraphDataset)

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test1"
    assert pose["pIC50"] == 5

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test2"
    assert pose["pIC50"] == 6
