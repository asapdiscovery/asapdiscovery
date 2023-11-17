import os

import pytest
from asapdiscovery.docking.openeye import POSITDocker


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking(docking_input_pair):
    docker = POSITDocker(use_omega=False)  # save compute
    results = docker.dock([docking_input_pair])
    assert len(results) == 1
    assert results[0].probability > 0.0


@pytest.mark.parametrize("omega_dense", [True, False])
@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking_omega(docking_input_pair, omega_dense):
    docker = POSITDocker(use_omega=True, omega_dense=omega_dense)
    results = docker.dock([docking_input_pair])
    assert len(results) == 1
    assert results[0].probability > 0.0


def test_docking_omega_dense_fails_no_omega(docking_input_pair):
    with pytest.raises(ValueError, match="Cannot use omega_dense without use_omega"):
        _ = POSITDocker(use_omega=False, omega_dense=True)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.parametrize("use_dask", [True, False])
def test_docking_dask(docking_input_pair, docking_input_pair_simple, use_dask):
    docker = POSITDocker(use_omega=False)  # save compute
    results = docker.dock(
        [docking_input_pair, docking_input_pair_simple], use_dask=use_dask
    )
    assert len(results) == 2
    assert results[0].probability > 0.0


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking_with_file_write(results_simple, tmp_path):
    docker = POSITDocker(use_omega=False)
    docker.write_docking_files(results_simple, tmp_path)
    sdf_path = tmp_path / "test_+_test2" / "docked.sdf"
    assert sdf_path.exists()
    pdb_path = tmp_path / "test_+_test2" / "docked_complex.pdb"
    assert pdb_path.exists()


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_multireceptor_docking(docking_multi_structure):
    assert len(docking_multi_structure.complexes) == 2
    docker = POSITDocker(use_omega=False)  # save compute
    results = docker.dock([docking_multi_structure])
    assert len(results) == 1
    assert results[0].input_pair.complex.target.target_name == "Mpro-x0354"
    assert results[0].probability > 0.0
