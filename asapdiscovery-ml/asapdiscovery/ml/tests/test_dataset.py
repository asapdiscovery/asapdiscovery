import pytest
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.dataset import DockedDataset, GroupedDockedDataset


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


def test_docked_dataset_from_complexes(complex_pdb):
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

    dd = DockedDataset.from_complexes([c1, c2])

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test1"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test2"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0


def test_docked_dataset_from_files(complex_pdb):
    dd = DockedDataset.from_files(
        str_fns=[complex_pdb, complex_pdb],
        compounds=[("test1", "test1"), ("test2", "test2")],
    )

    assert len(dd) == 2

    it = iter(dd)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test1"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == compound_id == "test2"
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0


def test_grouped_docked_dataset_from_complexes(complex_pdb):
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

    ds = GroupedDockedDataset.from_complexes([c1, c2])

    assert len(ds) == 1

    compound_id, pose_list = next(iter(ds))
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


def test_grouped_docked_dataset_from_files(complex_pdb):
    ds = GroupedDockedDataset.from_files(
        str_fns=[complex_pdb, complex_pdb],
        compounds=[("test1", "test"), ("test2", "test")],
        ignore_h=False,
    )

    assert len(ds) == 1

    compound_id, pose_list = next(iter(ds))
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
