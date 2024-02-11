import os

import pytest
from asapdiscovery.simulation.simulate import VanillaMDSimulator


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
        vs = VanillaMDSimulator(
            num_steps=1,
            equilibration_steps=1,
            output_dir=tmp_path,
            truncate_steps=False,
            rmsd_restraint=True,
            rmsd_restraint_indices=[1, 2, 3],
            rmsd_restraint_type="CA",
        )
