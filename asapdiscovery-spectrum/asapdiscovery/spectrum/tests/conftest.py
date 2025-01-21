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


@pytest.fixture()
def protein_apo_path():
    return fetch_test_file("Mpro-YP_009725301_AFold_processed.pdb")


@pytest.fixture()
def all_structure_dir_fns():
    return [
        "structure_dir/Mpro-x0354_0A_bound.pdb",
        "structure_dir/Mpro-x1002_0A_bound.pdb",
    ]


@pytest.fixture()
def structure_dir(all_structure_dir_fns):
    all_paths = [fetch_test_file(f) for f in all_structure_dir_fns]
    return all_paths[0].parent, all_paths
