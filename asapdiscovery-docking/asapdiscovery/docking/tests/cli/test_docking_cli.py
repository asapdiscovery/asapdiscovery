import os
import traceback
from mock import patch
import pytest
from asapdiscovery.docking.cli import docking as cli
from asapdiscovery.simulation.simulate_v2 import _SIMULATOR_TRUNCATE_STEPS
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.parametrize("subcommand", ["large-scale", "small-scale"])
def test_project_support_docking_cli_fragalysis(
    ligand_file, mpro_frag_dir, tmp_path, subcommand
):
    runner = CliRunner()

    frag_parent_dir, _ = mpro_frag_dir

    args = [
        subcommand,
        "--target",
        "SARS-CoV-2-Mpro",
        "--ligands",
        ligand_file,
        "--fragalysis-dir",
        frag_parent_dir,
        "--posit-confidence-cutoff",
        0,
        "--output-dir",
        tmp_path,
    ]

    if (
        subcommand == "small-scale"
    ):  # turn off dask cuda overrides for CI runners which lack GPUs
        args.extend(["--no-allow-dask-cuda"])

    result = runner.invoke(cli, args)
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.parametrize("subcommand", ["large-scale", "small-scale"])
def test_project_support_docking_cli_structure_directory_dask(
    ligand_file, structure_dir, tmp_path, subcommand
):
    runner = CliRunner()

    struct_dir, _ = structure_dir

    args = [
        subcommand,
        "--target",
        "SARS-CoV-2-Mpro",
        "--ligands",
        ligand_file,
        "--structure-dir",
        struct_dir,
        "--use-dask",  # add dask
        "--posit-confidence-cutoff",
        0,
        "--output-dir",
        tmp_path,
    ]

    if (
        subcommand == "small-scale"
    ):  # turn off dask cuda overrides for CI runners which lack GPUs
        args.extend(["--no-allow-dask-cuda"])

    result = runner.invoke(cli, args)
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.parametrize("subcommand", ["large-scale", "small-scale"])
def test_project_support_docking_cli_structure_directory_du_cache_dask(
    ligand_file, structure_dir, du_cache, tmp_path, subcommand
):
    runner = CliRunner()

    struct_dir, _ = structure_dir
    du_cache_dir, _ = du_cache

    args = [
        subcommand,
        "--target",
        "SARS-CoV-2-Mpro",
        "--ligands",
        ligand_file,
        "--structure-dir",
        struct_dir,
        "--use-dask",
        "--posit-confidence-cutoff",
        0,
        "--cache-dir",
        du_cache_dir,
        "--output-dir",
        tmp_path,
    ]

    if (
        subcommand == "small-scale"
    ):  # turn off dask cuda overrides for CI runners which lack GPUs
        args.extend(["--no-allow-dask-cuda"])

    result = runner.invoke(cli, args)
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.parametrize("subcommand", ["large-scale", "small-scale"])
def test_project_support_docking_cli_pdb_file_dask(
    ligand_file, pdb_file, tmp_path, subcommand
):
    runner = CliRunner()

    args = [
        subcommand,
        "--target",
        "SARS-CoV-2-Mpro",
        "--ligands",
        ligand_file,
        "--pdb-file",
        pdb_file,
        "--use-dask",
        "--posit-confidence-cutoff",
        0,
        "--output-dir",
        tmp_path,
    ]

    if (
        subcommand == "small-scale"
    ):  # turn off dask cuda overrides for CI runners which lack GPUs
        args.extend(["--no-allow-dask-cuda"])

    result = runner.invoke(cli, args)
    assert click_success(result)


@patch("asapdiscovery.simulation.simulate_v2._SIMULATOR_TRUNCATE_STEPS", False)
@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_small_scale_docking_md(ligand_file, pdb_file, tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "small-scale",
            "--target",
            "SARS-CoV-2-Mpro",
            "--ligands",
            ligand_file,
            "--pdb-file",
            pdb_file,
            "--use-dask",
            "--no-allow-dask-cuda",
            "--posit-confidence-cutoff",
            0,
            "--output-dir",
            tmp_path,
            "--md",
            "--md-steps",
            1,
            "--md-openmm-platform",
            "CPU",
        ],
    )
    # check that the number of steps was set to 1
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_cross_docking_cli_structure_directory_du_cache(
    ligand_file, structure_dir, du_cache, tmp_path
):
    runner = CliRunner()

    struct_dir, _ = structure_dir
    du_cache_dir, _ = du_cache

    result = runner.invoke(
        cli,
        [
            "cross-docking",
            "--target",
            "SARS-CoV-2-Mpro",
            "--ligands",
            ligand_file,
            "--structure-dir",
            struct_dir,
            "--use-dask",
            "--cache-dir",
            du_cache_dir,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)
