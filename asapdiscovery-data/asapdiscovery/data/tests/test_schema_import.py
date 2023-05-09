from asapdiscovery.data.schema import ExperimentalCompoundData, Target, Ligand
from asapdiscovery.data.testing.test_resources import fetch_test_file
import pytest


def test_classes():
    compound = ExperimentalCompoundData(compound_id="Test")
    assert compound.compound_id == "Test"


@pytest.fixture
def target_files():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    oedu = fetch_test_file(
        "Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.oedu"
    )
    pdb = fetch_test_file(
        "Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb"
    )
    return sdf, oedu, pdb


def test_ligand():
    l = Ligand(smiles="CC", id="XX112233", source="test", vc_id_postera="XX112233")

def test_ligand_invalid_smiles():
    with pytest.raises(ValueError):
        l = Ligand(smiles="GG", id="XX112233", source="test", vc_id_postera="XX112233")

def test_ligand_from_sdf(target_files):
    sdf, _, _ = target_files
    l = Ligand.from_sdf(sdf)
    assert l.smiles == "CC"


def test_target_from_pdb(target_files):
    _, pdb_fn = target_files

    target = Target.from_pdb(pdb_fn, "test")

    assert target.id == "test"
    assert target.chain == "A"
