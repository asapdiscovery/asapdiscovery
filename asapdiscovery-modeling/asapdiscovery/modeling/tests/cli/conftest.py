import pytest

from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture
def pdb_file():
    return fetch_test_file("Mpro-P2660_0A_bound_oe_processed.pdb")


@pytest.fixture(scope="session")
def all_mpro_fns():
    return [
        "metadata.csv",
        "aligned/Mpro-x0354_0A/Mpro-x0354_0A_bound.pdb",
        "aligned/Mpro-x1002_0A/Mpro-x1002_0A_bound.pdb",
    ]


@pytest.fixture(scope="session")
def mpro_frag_dir(all_mpro_fns):
    all_paths = [fetch_test_file(f"frag_factory_test/{fn}") for fn in all_mpro_fns]
    return all_paths[0].parent, all_paths


@pytest.fixture(scope="session")
def all_structure_dir_fns():
    return [
        "structure_dir/Mpro-x0354_0A_bound.pdb",
        "structure_dir/Mpro-x1002_0A_bound.pdb",
    ]


@pytest.fixture(scope="session")
def structure_dir(all_structure_dir_fns):
    all_paths = [fetch_test_file(f) for f in all_structure_dir_fns]
    return all_paths[0].parent, all_paths


@pytest.fixture(scope="session")
def du_cache_files():
    return ["du_cache/Mpro-x0354_0A_bound.oedu", "du_cache/Mpro-x1002_0A_bound.oedu"]


@pytest.fixture(scope="session")
def du_cache(du_cache_files):
    all_paths = [fetch_test_file(f) for f in du_cache_files]
    return all_paths[0].parent, all_paths


@pytest.fixture(scope="session")
def json_cache():
    """A mock json cache of prepared proteins"""
    return fetch_test_file("protein_json_cache/Mpro-x0354_0A_bound.json")
