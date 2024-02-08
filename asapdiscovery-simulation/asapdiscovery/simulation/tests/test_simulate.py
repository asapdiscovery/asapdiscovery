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
