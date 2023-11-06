from pathlib import Path

import pytest
from asapdiscovery.data.schema_v2.complex import PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import DockingInputPair
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.docking_v2 import POSITDocker


@pytest.fixture(scope="session")
def local_path(request):
    try:
        return request.config.getoption("--local_path")
    except ValueError:
        return None


# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture(scope="session")
def output_dir(tmp_path_factory, local_path):
    if type(local_path) is not str:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path = Path(local_path)
        local_path.mkdir(exist_ok=True)
        assert local_path.exists()
        return local_path


@pytest.fixture(scope="session")
def ligand():
    return Ligand.from_sdf(
        fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"), compound_name="test"
    )


@pytest.fixture(scope="session")
def ligand_simple():
    return Ligand.from_smiles("CCCOCO", compound_name="test2")


@pytest.fixture(scope="session")
def prepped_complex():
    return PreppedComplex.from_oedu_file(
        fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu"),
        ligand_kwargs={"compound_name": "test"},
        target_kwargs={"target_name": "test"},
    )


@pytest.fixture(scope="session")
def prepped_complexes():
    # A list of PreppedComplex objects

    # TODO: add another prepped complex to AWS
    return PreppedComplex.from_oedu_file(
        fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu"),
        ligand_kwargs={"compound_name": "test"},
        target_kwargs={"target_name": "test"},
    )


@pytest.fixture(scope="session")
def docking_input_pair(ligand, prepped_complex):
    return DockingInputPair(complex=prepped_complex, ligand=ligand)


@pytest.fixture(scope="session")
def docking_input_pair_simple(ligand_simple, prepped_complex):
    return DockingInputPair(complex=prepped_complex, ligand=ligand_simple)


@pytest.fixture(scope="session")
def results(docking_input_pair):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair])
    return results


@pytest.fixture(scope="session")
def results_simple(docking_input_pair_simple):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair_simple])
    return results


@pytest.fixture(scope="session")
def results_multi(results, results_simple):
    return results + results_simple
