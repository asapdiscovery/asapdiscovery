import pytest
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def sdf_file():
    return fetch_test_file("Mpro_combined_labeled.sdf")


@pytest.fixture(scope="session")
def smi_file():
    return fetch_test_file("Mpro_combined_labeled.smi")


def test_molfile_factory_sdf(sdf_file):
    molfile = MolFileFactory(filename=sdf_file)
    ligands = molfile.load()
    assert len(ligands) == 576


def test_molfile_factory_smi(smi_file):
    molfile = MolFileFactory(filename=smi_file)
    ligands = molfile.load()
    assert len(ligands) == 556
