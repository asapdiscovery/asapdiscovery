from pathlib import Path

import pytest
from asapdiscovery.data.backend.openeye import (
    load_openeye_pdb,
    load_openeye_sdf,
    load_openeye_sdfs,
    load_openeye_smi,
    oe_smiles_roundtrip,
)
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture
def pdb_file():
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb")


@pytest.fixture
def smi_file():
    return fetch_test_file("Mpro_combined_labeled.smi")


@pytest.fixture
def sdf_file():
    return fetch_test_file("Mpro_combined_labeled.sdf")


@pytest.fixture
def problematic_smiles():
    # https://github.com/rdkit/rdkit/discussions/3609
    problematic_RDKit_SMILES = "C[C@H](CCCC(C)(C)O)[C@H]1CC[C@H]2\\C(CCC[C@]12C)=C\\C=C/3C[C@@H](O)[C@H](OCCCO)[C@H](O)C/3=C"
    return problematic_RDKit_SMILES


def test_smi_roundtrip(problematic_smiles):
    new = oe_smiles_roundtrip(problematic_smiles)
    assert new != problematic_smiles


def test_load_pdb(pdb_file):
    _ = load_openeye_pdb(Path(pdb_file))
    # try also with str
    _ = load_openeye_pdb(str(pdb_file))


def test_load_smi(smi_file):
    mols = load_openeye_smi(Path(smi_file))
    assert len(mols) == 556
    # try also with str
    mols = load_openeye_smi(str(smi_file))
    assert len(mols) == 556


def test_load_sdf(sdf_file):
    _ = load_openeye_sdf(Path(sdf_file))
    _ = load_openeye_sdf(str(sdf_file))


def test_load_sdfs(sdf_file):
    oemols = load_openeye_sdfs(Path(sdf_file))
    assert len(oemols) == 576
    oemols = load_openeye_sdfs(str(sdf_file))
    assert len(oemols) == 576


def test_ligand_sdf(moonshot_sdf, multipose_ligand, sdf_file):
    single_conf = load_openeye_sdf(moonshot_sdf)
    assert single_conf.NumConfs() == 1
    multiconf = load_openeye_sdf(multipose_ligand)

    assert multiconf.NumConfs() == 50

    # right now `load_openeye_sdf` returns the first molecule in a multi-molecule sdf
    mol = load_openeye_sdf(sdf_file)
    assert mol.NumConfs() == 1


def test_sd_tag_processing(moonshot_sdf, multipose_ligand):
    from asapdiscovery.data.backend.openeye import get_SD_data, set_SD_data

    single_conf = load_openeye_sdf(moonshot_sdf)
    assert get_SD_data(single_conf) == {}

    multiconf = load_openeye_sdf(multipose_ligand)

    simple_data = {"test": "value"}
    set_SD_data(single_conf, simple_data)
    assert get_SD_data(single_conf)["test"] == ["value"]

    set_SD_data(multiconf, simple_data)
    assert get_SD_data(multiconf)["test"] == ["value"] * multiconf.NumConfs()

    with pytest.raises(ValueError):
        set_SD_data(single_conf, {"test": ["value", "value2"]})

    with pytest.raises(ValueError):
        set_SD_data(single_conf, {"test": []})

    with pytest.raises(ValueError):
        set_SD_data(multiconf, {"test": ["value", "value2"]})
