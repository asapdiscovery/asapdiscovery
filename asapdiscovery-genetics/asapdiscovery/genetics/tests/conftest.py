import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def protein_path():
    return fetch_test_file("Mpro-P2660_0A_bound.pdb")


@pytest.fixture(scope="session")
def blast_xml_path():
    return fetch_test_file("SARS_blast_results.xml")


@pytest.fixture(scope="session")
def blast_csv_path():
    return fetch_test_file("SARS_blast_results.csv")
