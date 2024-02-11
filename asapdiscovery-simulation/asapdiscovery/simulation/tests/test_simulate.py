import os

import pydantic
import pytest
from asapdiscovery.simulation.simulate import VanillaMDSimulator
from openmm import unit


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_actual_simulation(results, tmp_path):
    vs = VanillaMDSimulator(
        num_steps=1, equilibration_steps=1, output_dir=tmp_path, truncate_steps=False
    )
    assert vs.num_steps == 1
    assert vs.equilibration_steps == 1
    simulation_results = vs.simulate(results)
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
