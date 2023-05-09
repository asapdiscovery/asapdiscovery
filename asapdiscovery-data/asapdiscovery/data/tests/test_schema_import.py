import pytest

from asapdiscovery.data.schema import ExperimentalCompoundData, Target
from asapdiscovery.data.testing.test_resources import fetch_test_file


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


def test_target_from_pdb():
    _, pdb_fn = target_files

    target = Target.from_pdb(pdb_fn, "test")

    assert target.id == "test"
    assert target.chain == "A"
