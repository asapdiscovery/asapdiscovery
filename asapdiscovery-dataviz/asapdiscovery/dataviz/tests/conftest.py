import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.openeye import POSITDockingResults
from asapdiscovery.simulation.simulate import SimulationResult


@pytest.fixture(scope="session")
def pose():
    pose = fetch_test_file("Mpro-P2660_0A_bound-prepped_ligand.sdf")
    return pose


@pytest.fixture(scope="session")
def protein():
    protein = fetch_test_file("Mpro-P2660_0A_bound-prepped_complex.pdb")
    return protein


@pytest.fixture(scope="session")
def top():
    top = fetch_test_file("example_traj_top.pdb")
    return top


@pytest.fixture(scope="session")
def traj():
    traj = fetch_test_file("example_traj.xtc")
    return traj


@pytest.fixture(scope="session")
def docking_results_file():
    results = fetch_test_file("docking_results.json")
    return [results]


@pytest.fixture(scope="session")
def docking_results_in_memory(docking_results_file):
    return [POSITDockingResults.from_json_file(docking_results_file[0])]


@pytest.fixture(scope="session")
def simulation_results(docking_results_in_memory):
    return [
        SimulationResult(
            input_docking_result=docking_results_in_memory[0],
            traj_path=fetch_test_file("example_traj.xtc"),
            minimized_pdb_path=fetch_test_file("example_traj_top.pdb"),
            final_pdb_path=fetch_test_file("example_traj_top.pdb"),
            success=True,
        )
    ]
