import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.openeye import POSITDockingResults
from asapdiscovery.simulation.szybki import SzybkiFreeformResult


@pytest.fixture(scope="session")
def ligand_path():
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")


@pytest.fixture(scope="session")
def results():
    res = POSITDockingResults.from_json_file(fetch_test_file("docking_results.json"))
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
