import os
import traceback
from unittest import mock

import pytest
from asapdiscovery.docking.docking import DockingResult
from asapdiscovery.simulation.simulate import SimulationResult, VanillaMDSimulator
from asapdiscovery.workflows.docking_workflows.cli import docking as cli
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
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
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
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
@pytest.mark.parametrize("subcommand", ["large-scale", "small-scale"])
def test_project_support_docking_cli_structure_directory(
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
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
@pytest.mark.parametrize("subcommand", ["large-scale", "small-scale"])
def test_project_support_docking_cli_structure_directory_du_cache(
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


@pytest.mark.skip(reason="Test is broken on GHA but should run locally")
@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
@pytest.mark.parametrize("subcommand", ["large-scale", "small-scale"])
def test_project_support_docking_cli_pdb_file(
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
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
@pytest.mark.skip(
    "Test is broken on GHA due to massive resource use but should run locally"
)
def test_small_scale_docking_md(ligand_file, pdb_file, tmp_path, simulation_results):
    runner = CliRunner()

    def _simulate_patch(
        self, inputs: list[DockingResult], **kwargs
    ) -> list[SimulationResult]:
        return [simulation_results]

    # NB: cannot use dask for below test as patch will not survive pickling and transfer to worker

    with mock.patch.object(VanillaMDSimulator, "_simulate", _simulate_patch):
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
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
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
            "--cache-dir",
            du_cache_dir,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_symexp_workflow(ligand_file, pdb_file, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "symexp-crystal-packing",
            "--target",
            "SARS-CoV-2-Mpro",
            "--ligands",
            ligand_file,
            "--pdb-file",
            pdb_file,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
@pytest.mark.skip()  # Test is broken on GHA but should run locally
def test_ligand_transfer_workflow(pdb_apo_file, pdb_file, tmp_path, simulation_results):
    runner = CliRunner()

    def _simulate_patch(
        self, inputs: list[DockingResult], **kwargs
    ) -> list[SimulationResult]:
        return [simulation_results]

    # NB: cannot use dask for below test as patch will not survive pickling and transfer to worker

    with mock.patch.object(VanillaMDSimulator, "_simulate", _simulate_patch):
        result = runner.invoke(
            cli,
            [
                "ligand-transfer-docking",
                "--target",
                "SARS-CoV-2-Mpro",
                "--ref-pdb-file",
                pdb_file,
                "--pdb-file",
                pdb_apo_file,
                "--output-dir",
                tmp_path,
                "--no-allow-dask-cuda",
                "--posit-confidence-cutoff",
                0,
                "--allow-final-clash",
                "--allow-retries",
                "--md",
                "--md-steps",
                1,
                "--md-openmm-platform",
                "CPU",
            ],
        )
    assert click_success(result)
