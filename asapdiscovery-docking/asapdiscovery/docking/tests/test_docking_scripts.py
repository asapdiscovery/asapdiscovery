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
@pytest.mark.timeout(400)
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("use_glob", [True, False])
@pytest.mark.script_launch_mode("subprocess")
def test_docking_base(script_runner, output_dir, docking_files_single, n, use_glob):
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
        f"{output_dir}",
        "-n",
        f"{n}",
    )
    assert ret.success


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.timeout(400)
@pytest.mark.parametrize("posit_method", POSIT_METHODS)
def test_posit_methods(
    script_runner,
    output_dir,
    docking_files_single,
    posit_method,
):
    sdf, oedu, _, _ = docking_files_single

    args = [
        "run-docking-oe",
        "-l",
        f"{sdf}",
        "-r",
        f"{oedu}",
        "-o",
        f"{output_dir / posit_method}",
        "-n",
        "1",
        "--posit_method",
        f"{posit_method}",
        "--omega",
    ]
    ret = script_runner.run(*args)
    assert ret.success


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.timeout(400)
@pytest.mark.parametrize("omega", [False, "--omega"])
@pytest.mark.parametrize("posit_method", POSIT_METHODS)
@pytest.mark.parametrize("ml", [False, ["--gat", "--schnet"]])
@pytest.mark.script_launch_mode("subprocess")
def test_docking_kwargs(
    script_runner,
    output_dir,
    docking_files_single,
    omega,
    posit_method,
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
        f"{output_dir}",
        "-n",
        "1",
        "--posit_method",
        f"{posit_method}",
    ]
    if omega:
        args.append(omega)

    if ml:
        args += ml
    ret = script_runner.run(*args)
    assert ret.success


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.timeout(500)
@pytest.mark.script_launch_mode("subprocess")
def test_single_target_docking(
    script_runner,
    output_dir,
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
        f"{output_dir}",
        "--viz-target",
        "SARS-CoV-2-Mpro",
        "--target",
        "SARS-CoV-2-Mpro",
        "--no-omega",
    ]
    ret = script_runner.run(args)
    assert ret.success


@pytest.mark.timeout(400)
@pytest.mark.parametrize("by_compound", ["--by_compound"])
@pytest.mark.script_launch_mode("subprocess")
def test_failing_kwargs(
    script_runner,
    output_dir,
    docking_files_single,
    by_compound,
):
    sdf, oedu, _, _ = docking_files_single

    args = [
        "run-docking-oe",
        "-l",
        f"{sdf}",
        "-r",
        f"{oedu}",
        "-o",
        f"{output_dir}",
        "-n",
        "1",
        by_compound,
    ]

    # should fail when specifying a single receptor and by_compound
    ret = script_runner.run(*args)
    assert not ret.success


