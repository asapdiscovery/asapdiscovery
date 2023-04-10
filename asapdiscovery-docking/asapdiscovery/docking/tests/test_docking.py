import pytest
import os
import subprocess


def finalizer_function():
    yield
    # clean up
    shutil.rmtree("./outputs")


@pytest.mark.timeout(200)
@pytest.mark.script_launch_mode("subprocess")
def test_docking(script_runner):
    os.makedirs("./outputs", exist_ok=True)
    ret = script_runner.run("run-docking-oe")
    assert ret.success
