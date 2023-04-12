import os
import shutil
import subprocess

import pytest


@pytest.fixture()
def make_output_dir_and_cleanup():
    # create output dir
    os.makedirs("./outputs", exist_ok=True)
    yield
    # clean up
    shutil.rmtree("./outputs")


@pytest.mark.timeout(200)
@pytest.mark.script_launch_mode("subprocess")
def test_docking(script_runner, make_output_dir_and_cleanup):
    ret = script_runner.run(
        "run-docking-oe",
        "-l",
        "input/Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf",
        "-r",
        "'sars_01_prepped_v2/*/*prepped_receptor_0.oedu'",
        "-o",
        "./outputs",
    )
    assert ret.success
    # can add other checks here
