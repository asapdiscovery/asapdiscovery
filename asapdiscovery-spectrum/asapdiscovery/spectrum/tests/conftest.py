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


@pytest.fixture(scope="session")
def fasta_alignment_path():
    return fetch_test_file("sars_alignment.fasta")


@pytest.fixture()
def protein_apo_path():
    return fetch_test_file("Mpro-YP_009725301_AFold_processed.pdb")


@pytest.fixture()
def protein_mers_path():
    return fetch_test_file("mers_8hut.pdb")


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


@pytest.fixture()
def all_cfold_dir_fns():
    return [
        "spectrum_dir/YP_009725295_1_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
        "spectrum_dir/YP_009725295_1_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb",
    ]


@pytest.fixture()
def cfold_dir(all_cfold_dir_fns):
    all_paths = [fetch_test_file(f) for f in all_cfold_dir_fns]
    return all_paths[0].parent, all_paths
