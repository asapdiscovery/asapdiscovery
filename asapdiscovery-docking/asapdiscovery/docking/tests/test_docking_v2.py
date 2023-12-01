import os

import pandas as pd
import pytest
from asapdiscovery.docking.openeye import POSITDocker, POSITDockingResults


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


@pytest.mark.parametrize("use_dask", [True, False])
def test_dock_and_save_with_check_results(docking_input_pair, use_dask, tmp_path):
    docker = POSITDocker(use_omega=False)  # save compute
    results = docker.dock_and_save(
        [docking_input_pair],
        use_dask=use_dask,
        output_dir=tmp_path,
    )
    results_df = POSITDockingResults.make_df_from_docking_results(results)

    # because of annoying floating point precision issues, we need to round the probabilities
    # instead of figuring out how that works, I'm just gonna manually save and load
    results_df.to_csv(tmp_path / "saved_docking_results.csv")
    results_df = pd.read_csv(tmp_path / "saved_docking_results.csv", index_col=0)

    sdf_path = tmp_path / "test_+_test" / "docked.sdf"
    pdb_path = tmp_path / "test_+_test" / "docked_complex.pdb"
    csv_path = tmp_path / "test_+_test" / "docking_results.csv"

    assert sdf_path.exists()
    assert pdb_path.exists()
    assert csv_path.exists()

    loaded_results = pd.read_csv(csv_path, index_col=0)
    loaded_results.to_csv("loaded_docking_results.csv")
    assert loaded_results.equals(results_df)

    assert docker.check_results_exist(docking_input_pair, tmp_path)

    assert docker.get_unfinished_results([docking_input_pair], tmp_path) == []

    # test to make sure that if we try to dock again, it doesn't do anything
    results = docker.dock_and_save(
        [docking_input_pair], use_dask=use_dask, output_dir=tmp_path
    )
    assert len(results) == 0

    # but if we tell it to overwrite, it does
    results = docker.dock_and_save(
        [docking_input_pair],
        use_dask=use_dask,
        output_dir=tmp_path,
        overwrite_existing=True,
    )
    assert len(results) == 1
