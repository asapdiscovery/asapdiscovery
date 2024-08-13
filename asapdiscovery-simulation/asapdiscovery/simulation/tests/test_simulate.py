import os
import traceback
from unittest import mock

import pytest
from asapdiscovery.docking.docking import DockingResult
from asapdiscovery.docking.openeye import POSITDockingResults
from asapdiscovery.simulation.cli import simulation as cli
from asapdiscovery.simulation.simulate import SimulationResult, VanillaMDSimulator
from click.testing import CliRunner
from openmm import unit


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.parametrize("use_dask", [True, False])
def test_actual_simulation(results, tmp_path, use_dask):
    vs = VanillaMDSimulator(
        num_steps=1,
        equilibration_steps=1,
        reporting_interval=1,
        output_dir=tmp_path,
        truncate_steps=False,
    )
    assert vs.num_steps == 1
    assert vs.equilibration_steps == 1
    simulation_results = vs.simulate(results, use_dask=use_dask, failure_mode="raise")
    assert simulation_results[0].success


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.parametrize("use_dask", [True, False])
def test_actual_simulation_disk(results_path, tmp_path, use_dask):
    vs = VanillaMDSimulator(
        num_steps=1,
        equilibration_steps=1,
        reporting_interval=1,
        output_dir=tmp_path,
        truncate_steps=False,
    )
    assert vs.num_steps == 1
    assert vs.equilibration_steps == 1
    simulation_results = vs.simulate(
        results_path,
        use_dask=use_dask,
        backend="disk",
        reconstruct_cls=POSITDockingResults,
        failure_mode="raise",
    )
    assert simulation_results[0].success


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_actual_simulation_paths(tyk2_protein, tmp_path, tyk2_lig):
    vs = VanillaMDSimulator(
        num_steps=1,
        equilibration_steps=1,
        reporting_interval=1,
        output_dir=tmp_path,
        truncate_steps=False,
    )
    assert vs.num_steps == 1
    assert vs.equilibration_steps == 1
    simulation_results = vs.simulate(
        [(tyk2_protein, tyk2_lig)], outpaths=[tmp_path], failure_mode="raise"
    )
    assert simulation_results[0].traj_path.exists()
    assert simulation_results[0].success


@pytest.mark.parametrize("restr_type", ["CA", "heavy"])
def test_rmsd_restraint(tmp_path, restr_type):
    vs = VanillaMDSimulator(
        num_steps=1,
        equilibration_steps=1,
        output_dir=tmp_path,
        truncate_steps=False,
        rmsd_restraint=True,
        rmsd_restraint_type=restr_type,
    )
    assert vs.num_steps == 1


def test_rmsd_restraint_fail(tmp_path):
    with pytest.raises(ValueError):
        VanillaMDSimulator(
            num_steps=1,
            equilibration_steps=1,
            output_dir=tmp_path,
            truncate_steps=False,
            rmsd_restraint=True,
            rmsd_restraint_type="fake_restraint",
        )


def test_rmsd_restraint_indices(tmp_path):
    vs = VanillaMDSimulator(
        num_steps=1,
        equilibration_steps=1,
        output_dir=tmp_path,
        truncate_steps=False,
        rmsd_restraint=True,
        rmsd_restraint_indices=[1, 2, 3],
    )
    assert vs.num_steps == 1
    assert vs.rmsd_restraint_indices == [1, 2, 3]


def test_rmsd_restraint_indices_mutex_type(tmp_path):
    with pytest.raises(ValueError):
        _ = VanillaMDSimulator(
            num_steps=1,
            equilibration_steps=1,
            output_dir=tmp_path,
            truncate_steps=False,
            rmsd_restraint=True,
            rmsd_restraint_indices=[1, 2, 3],
            rmsd_restraint_type="CA",
        )


def test_properties(tmp_path):
    vs = VanillaMDSimulator(
        num_steps=1000,
        equilibration_steps=1000,
        reporting_interval=5,
        output_dir=tmp_path,
    )
    assert vs.num_steps == 1000
    assert vs.equilibration_steps == 1000
    assert vs.output_dir == tmp_path
    assert vs.n_frames == 200
    assert vs.total_simulation_time == unit.Quantity(4000, unit.femtosecond)
    assert vs.frames_per_ns == 50000.0


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_multi_use(results, tmp_path):
    vs = VanillaMDSimulator(
        num_steps=1,
        equilibration_steps=1,
        reporting_interval=1,
        output_dir=tmp_path,
        truncate_steps=False,
    )
    assert vs.num_steps == 1
    assert vs.equilibration_steps == 1
    simulation_results = vs.simulate(results, failure_mode="raise")
    assert simulation_results[0].success

    simulation_results_parallel = vs.simulate(
        results, use_dask=True, failure_mode="raise"
    )

    assert simulation_results_parallel[0].success


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_simulation_cli(
    tyk2_protein,
    tmp_path,
    tyk2_lig,
    simulation_results,
):
    runner = CliRunner()

    def _simulate_patch(
        self, inputs: list[DockingResult], **kwargs
    ) -> list[SimulationResult]:
        return [simulation_results]

    # NB: cannot use dask for below test as patch will not survive pickling and transfer to worker

    with mock.patch.object(VanillaMDSimulator, "_simulate", _simulate_patch):
        args = [
            "vanilla-md",
            "--ligands",
            tyk2_lig,
            "--pdb-file",
            tyk2_protein,
            "--output-dir",
            tmp_path,
        ]
        result = runner.invoke(cli, args)
    assert click_success(result)
