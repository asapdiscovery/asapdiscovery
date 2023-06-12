import os

import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def docking_files_single():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    oedu = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.oedu")
    pdb = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb")
    oedu_glob = os.path.join(os.path.dirname(oedu), "*.oedu")
    return sdf, oedu, oedu_glob, pdb


@pytest.fixture()
def outputs(tmp_path):
    """Creates outputs directory in temp location and returns path"""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS")
@pytest.mark.timeout(400)
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("use_glob", [True, False])
@pytest.mark.script_launch_mode("subprocess")
def test_docking_base(script_runner, outputs, docking_files_single, n, use_glob):
    sdf, oedu, oedu_glob, _ = docking_files_single
    if use_glob:
        oedu = oedu_glob

    ret = script_runner.run(
        "run-docking-oe",
        "-l",
        f"{sdf}",
        "-r",
        f"{oedu}",
        "-o",
        f"{outputs}",
        "-n",
        f"{n}",
    )
    assert ret.success

@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS")
@pytest.mark.timeout(400)
@pytest.mark.parametrize("omega", [False, "--omega"])
@pytest.mark.parametrize("by_compound", [False, "--by_compound"])
@pytest.mark.parametrize("hybrid", [False, "--hybrid"])
@pytest.mark.parametrize("ml", [False, ["--gat", "--schnet"]])
@pytest.mark.script_launch_mode("subprocess")
def test_docking_kwargs(
    script_runner,
    outputs,
    docking_files_single,
    omega,
    by_compound,
    hybrid,
    ml,
):
    sdf, oedu, _, _ = docking_files_single

    args = [
        "run-docking-oe",
        "-l",
        f"{sdf}",
        "-r",
        f"{oedu}",
        "-o",
        f"{outputs}",
        "-n",
        "1",
    ]
    if omega:
        args.append(omega)

    if hybrid:
        args.append(hybrid)

    if ml:
        args += ml

    if by_compound:
        # should fail when specifying a single receptor and by_compound
        args.append(by_compound)
        ret = script_runner.run(*args)
        assert not ret.success
    else:
        ret = script_runner.run(*args)
        assert ret.success
