from datetime import datetime
from functools import reduce

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
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset, GroupedDockedDataset


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

    exp_data = {"test1": {"pIC50": 5}, "test2": {"pIC50": 6}}

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
    assert len(pose_list["poses"]) == 2

    pose = pose_list["poses"][0]
    assert pose_list["compound"] == ("test1", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0

    pose = pose_list["poses"][1]
    assert pose["compound"] == ("test2", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0


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

    exp_data = {"test": {"pIC50": 5}}

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
    assert len(pose_list["poses"]) == 2

    pose = pose_list["poses"][0]
    assert pose["compound"] == ("test1", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0
    assert pose["pIC50"] == 5

    pose = pose_list["poses"][1]
    assert pose["compound"] == ("test2", "test")
    assert pose["pos"].shape[0] == len(pose["z"]) == len(pose["lig"])
    assert pose["pos"].shape[0] > 0
    assert pose["pIC50"] == 5


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

    exp_data = {"test1": {"pIC50": 5}, "test2": {"pIC50": 6}}

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


def test_random_splitting_no_seed(ligand_sdf):
    ligands = [Ligand.from_sdf(ligand_sdf, compound_name=f"test{i}") for i in range(10)]
    dd = DatasetConfig(ds_type=DatasetType.graph, input_data=ligands).build()

    splitter = DatasetSplitterConfig(split_type=DatasetSplitterType.random)

    # Split a couple times to check randomness
    all_splits = [splitter.split(dd) for _ in range(10)]

    # Make sure that each split in each splitting instance is the right length
    try:
        assert all([len(sp[0]) == 8 for sp in all_splits])
    except AssertionError as e:
        print([(len(sp[0]), len(sp[1]), len(sp[2])) for sp in all_splits], flush=True)
        print(
            [[pose["compound"][1] for (_, pose) in d] for d in next(iter(all_splits))],
            flush=True,
        )
        raise e
    assert all([len(sp[1]) == 1 for sp in all_splits])
    assert all([len(sp[2]) == 1 for sp in all_splits])

    # Make sure that each split keeps all compounds and there are no duplicates
    for sp in all_splits:
        sp_compounds = set()
        for d in sp:
            sp_compounds.update([pose["compound"][1] for (_, pose) in d])
        assert len(sp_compounds) == 10

    # Make sure that not all splits are the same
    all_train_compound_ids = [
        {pose["compound"][1] for (_, pose) in sp[0]} for sp in all_splits
    ]
    shared_train_compound_ids = reduce(
        lambda s1, s2: s1.intersection(s2), all_train_compound_ids
    )
    assert len(shared_train_compound_ids) < 8


def test_random_splitting_set_seed(ligand_sdf):
    ligands = [Ligand.from_sdf(ligand_sdf, compound_name=f"test{i}") for i in range(10)]
    dd = DatasetConfig(ds_type=DatasetType.graph, input_data=ligands).build()

    splitter = DatasetSplitterConfig(
        split_type=DatasetSplitterType.random, rand_seed=42
    )

    # Split a couple times to check randomness
    all_splits = [splitter.split(dd) for _ in range(10)]

    # Make sure that each split in each splitting instance is the right length
    try:
        assert all([len(sp[0]) == 8 for sp in all_splits])
    except AssertionError as e:
        print([(len(sp[0]), len(sp[1]), len(sp[2])) for sp in all_splits], flush=True)
        print(
            [[pose["compound"][1] for (_, pose) in d] for d in next(iter(all_splits))],
            flush=True,
        )
        raise e
    assert all([len(sp[1]) == 1 for sp in all_splits])
    assert all([len(sp[2]) == 1 for sp in all_splits])

    # Make sure that each split keeps all compounds and there are no duplicates
    for sp in all_splits:
        sp_compounds = set()
        for d in sp:
            sp_compounds.update([pose["compound"][1] for (_, pose) in d])
        assert len(sp_compounds) == 10

    # Make sure that not all splits are the same
    all_train_compound_ids = [
        {pose["compound"][1] for (_, pose) in sp[0]} for sp in all_splits
    ]
    shared_train_compound_ids = reduce(
        lambda s1, s2: s1.intersection(s2), all_train_compound_ids
    )
    assert len(shared_train_compound_ids) == 8


def test_temporal_splitting(ligand_sdf):
    ligands = [Ligand.from_sdf(ligand_sdf, compound_name=f"test{i}") for i in range(10)]

    # Add date_created field for each pose
    exp_data = {
        f"test{i}": {"date_created": datetime(2023, 1, i + 1)} for i in range(10)
    }

    dd = DatasetConfig(
        ds_type=DatasetType.graph, input_data=ligands, exp_data=exp_data
    ).build()

    splitter = DatasetSplitterConfig(split_type=DatasetSplitterType.temporal)

    ds_train, ds_val, ds_test = splitter.split(dd)

    assert {pose["compound"][1] for (_, pose) in ds_train} == {
        f"test{i}" for i in range(8)
    }
    assert {pose["compound"][1] for (_, pose) in ds_val} == {"test8"}
    assert {pose["compound"][1] for (_, pose) in ds_test} == {"test9"}
