import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.openeye import POSITDockingResults
from asapdiscovery.simulation.simulate import SimulationResult
from asapdiscovery.simulation.szybki import SzybkiFreeformResult


@pytest.fixture(scope="session")
def ligand_path():
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")


@pytest.fixture(scope="session")
def protein_path():
    return fetch_test_file("Mpro-x0071_0A_ERI-UCB-8c6b7d0d-1.pdb")


@pytest.fixture(scope="session")
def tyk2_protein():
    return fetch_test_file("tyk2_protein.pdb")


@pytest.fixture(scope="session")
def tyk2_lig():
    return fetch_test_file("tyk2_one_lig.sdf")


@pytest.fixture(scope="session")
def results_path():
    return [fetch_test_file("docking_results.json")]


@pytest.fixture(scope="session")
def results(results_path):
    res = POSITDockingResults.from_json_file(results_path[0])
    return [res]


@pytest.fixture(scope="session")
def szybki_results():
    res = SzybkiFreeformResult(
        ligand_id="bleh",
        szybki_global_strain=1.1,
        szybki_local_strain=1,
        szybki_conformer_strain=0.1,
    )
    return res


@pytest.fixture()
def simulation_results(results):
    return SimulationResult(
        input_docking_result=results[0],
        traj_path=fetch_test_file("example_traj.xtc"),
        minimized_pdb_path=fetch_test_file("example_traj_top.pdb"),
        final_pdb_path=fetch_test_file("example_traj_top.pdb"),
        success=True,
    )
