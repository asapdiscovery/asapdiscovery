from pathlib import Path

import pytest
from asapdiscovery.data.openeye import (
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
    _ = load_openeye_smi(Path(smi_file))
    # try also with str
    _ = load_openeye_smi(str(smi_file))


def test_load_sdf(sdf_file):
    _ = load_openeye_sdf(Path(sdf_file))
    _ = load_openeye_sdf(str(sdf_file))


def test_load_sdfs(sdf_file):
    oemols = load_openeye_sdfs(Path(sdf_file))
    assert len(oemols) == 576
    oemols = load_openeye_sdfs(str(sdf_file))
    assert len(oemols) == 576