import os

import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.docking import POSIT_METHODS


@pytest.fixture(scope="session")
def docking_files_single():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    oedu = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.oedu")
    oedu_glob = os.path.join(os.path.dirname(oedu), "*.oedu")
    pdb = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb")
    return sdf, oedu, oedu_glob, pdb


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.timeout(500)
@pytest.mark.script_launch_mode("subprocess")
def test_single_target_docking(
    script_runner,
    tmp_path,
    docking_files_single,
):
    sdf, oedu, _, pdb = docking_files_single
    args = [
        "dock-small-scale-e2e",
        "-m",
        f"{sdf}",
        "-r",
        f"{pdb}",
        "-o",
        f"{tmp_path}",
        "--viz-target",
        "SARS-CoV-2-Mpro",
        "--target",
        "SARS-CoV-2-Mpro",
        "--no-omega",
    ]
    ret = script_runner.run(args)
    assert ret.success
