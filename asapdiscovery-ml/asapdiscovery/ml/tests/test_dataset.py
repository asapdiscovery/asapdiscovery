import pytest
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.experimental import ExperimentalCompoundData
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset, GroupedDockedDataset


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


@pytest.fixture(scope="session")
def ligand_sdf():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    return sdf


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
    assert len(pose_list["poses"]) == 2

    pose = pose_list["poses"][0]
    assert pose["compound"] == ("test1", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0

    pose = pose_list["poses"][1]
    assert pose["compound"] == ("test2", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0


def test_grouped_docked_dataset_from_files(complex_pdb):
    ds = GroupedDockedDataset.from_files(
        str_fns=[complex_pdb, complex_pdb],
        compounds=[("test1", "test"), ("test2", "test")],
        ignore_h=False,
    )

    assert len(ds) == 1

    compound_id, pose_list = next(iter(ds))
    assert compound_id == "test"
    assert len(pose_list["poses"]) == 2

    pose = pose_list["poses"][0]
    assert pose["compound"] == ("test1", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0

    pose = pose_list["poses"][1]
    assert pose["compound"] == ("test2", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0


def test_graph_dataset_from_ligands(ligand_sdf, tmp_path):
    lig1 = Ligand.from_sdf(ligand_sdf, compound_name="test1")
    lig2 = Ligand.from_sdf(ligand_sdf, compound_name="test2")

    ds = GraphDataset.from_ligands([lig1, lig2])

    assert len(ds) == 2

    it = iter(ds)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test1"

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test2"


def test_graph_dataset_from_exp_compounds(ligand_sdf, tmp_path):
    lig = Ligand.from_sdf(ligand_sdf, compound_name="test")

    exp_data = {"pIC50": 5.1, "pIC50_range": 0, "pIC50_stderr": 0.3}

    exp1 = ExperimentalCompoundData(
        compound_id="test1", smiles=lig.smiles, experimental_data=exp_data
    )
    exp2 = ExperimentalCompoundData(
        compound_id="test2", smiles=lig.smiles, experimental_data=exp_data
    )

    ds = GraphDataset.from_exp_compounds([exp1, exp2])

    assert len(ds) == 2

    it = iter(ds)
    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test1"
    assert pose["pIC50"] == 5.1
    assert pose["pIC50_range"] == 0
    assert pose["pIC50_stderr"] == 0.3

    (xtal_id, compound_id), pose = next(it)
    assert xtal_id == "NA"
    assert compound_id == "test2"
    assert pose["pIC50"] == 5.1
    assert pose["pIC50_range"] == 0
    assert pose["pIC50_stderr"] == 0.3
