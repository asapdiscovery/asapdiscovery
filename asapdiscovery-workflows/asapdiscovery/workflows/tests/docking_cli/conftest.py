import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.simulation.simulate import SimulationResult


@pytest.fixture
def ligand_file():
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")


@pytest.fixture
def pdb_file():
    return fetch_test_file("Mpro-P2660_0A_bound_oe_processed.pdb")


@pytest.fixture()
def pdb_apo_file():
    return fetch_test_file("Mpro-YP_009725301_AFold_processed.pdb")


@pytest.fixture()
def all_mpro_fns():
    return [
        "metadata.csv",
        "aligned/Mpro-x0354_0A/Mpro-x0354_0A_bound.pdb",
        "aligned/Mpro-x1002_0A/Mpro-x1002_0A_bound.pdb",
    ]


@pytest.fixture()
def mpro_frag_dir(all_mpro_fns):
    all_paths = [fetch_test_file(f"frag_factory_test/{fn}") for fn in all_mpro_fns]
    return all_paths[0].parent, all_paths


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
def du_cache_files():
    return ["du_cache/Mpro-x0354_0A_bound.oedu", "du_cache/Mpro-x1002_0A_bound.oedu"]


@pytest.fixture()
def du_cache(du_cache_files):
    all_paths = [fetch_test_file(f) for f in du_cache_files]
    return all_paths[0].parent, all_paths


@pytest.fixture()
def simulation_results(results):
    return SimulationResult(
        input_docking_result=results[0],
        traj_path=fetch_test_file("example_traj.xtc"),
        minimized_pdb_path=fetch_test_file("example_traj_top.pdb"),
        final_pdb_path=fetch_test_file("example_traj_top.pdb"),
        success=True,
    )
