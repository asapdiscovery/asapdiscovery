import pytest
from asapdiscovery.data.postera.manifold_data_validation import TargetTags


@pytest.mark.script_launch_mode("subprocess")
def test_traj_to_viz(tmp_path, script_runner, traj, top):
    ret = script_runner.run(
        "traj-to-viz",
        "--target",
        "SARS-CoV-2-Mpro",
        "--traj",
        traj,
        "--top",
        top,
        "--out",
        f"{tmp_path}/viz.gif",
    )
    assert ret.success


@pytest.mark.parametrize("target", sorted(TargetTags.get_values()))
@pytest.mark.script_launch_mode("subprocess")
def test_pose_to_viz(tmp_path, script_runner, target, pose, protein):
    ret = script_runner.run(
        "pose-to-viz",
        "--target",
        target,
        "--pose",
        pose,
        "--protein",
        protein,
        "--out",
        f"{tmp_path}/viz.html",
    )
    assert ret.success


@pytest.mark.parametrize("target", sorted(TargetTags.get_values()))
@pytest.mark.script_launch_mode("subprocess")
def test_pose_to_viz_pymol(tmp_path, script_runner, target, protein):
    ret = script_runner.run(
        "pose-to-viz-pymol",
        "--target",
        target,
        "--complex",
        protein,
        "--out",
        f"{tmp_path}/viz.pse",
    )
    assert ret.success
