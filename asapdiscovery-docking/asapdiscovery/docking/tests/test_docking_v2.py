import os

import dask
import pytest
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import DockingInputPair
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.docking_v2 import POSITDocker


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
    cmplx = Complex.from_pdb(
        fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb"),
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    return PreppedComplex.from_complex(cmplx)


@pytest.fixture(scope="session")
def docking_input_pair(ligand, prepped_complex):
    return DockingInputPair(complex=prepped_complex, ligand=ligand)


@pytest.fixture(scope="session")
def docking_input_pair_simple(ligand_simple, prepped_complex):
    return DockingInputPair(complex=prepped_complex, ligand=ligand_simple)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking(docking_input_pair):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair])
    assert len(results) == 1
    assert results[0].probability > 0.0


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking_dask(docking_input_pair):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair], use_dask=True)
    assert len(results) == 1
    assert dask.is_dask_collection(results[0])


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking_with_file_write(docking_input_pair_simple, tmp_path):
    docker = POSITDocker(write_files=True, output_dir=tmp_path)
    results = docker.dock([docking_input_pair_simple])
    assert results[0].probability > 0.0
    sdf_path = tmp_path / "test2" / "docked.sdf"
    assert sdf_path.exists()
    pdb_path = tmp_path / "test2" / "docked_complex.pdb"
    assert pdb_path.exists()
