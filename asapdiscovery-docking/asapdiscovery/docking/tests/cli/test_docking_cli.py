import os
import traceback

import pytest
from asapdiscovery.docking.cli import docking as cli
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
def test_large_docking_cli_fragalysis(ligand_file, mpro_frag_dir, tmp_path):
    runner = CliRunner()

    frag_parent_dir, _ = mpro_frag_dir

    result = runner.invoke(
        cli,
        [
            "large-scale",
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
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_large_docking_cli_structure_directory_dask(
    ligand_file, structure_dir, tmp_path
):
    runner = CliRunner()

    struct_dir, _ = structure_dir

    result = runner.invoke(
        cli,
        [
            "large-scale",
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
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_large_docking_cli_structure_directory_du_cache(
    ligand_file, structure_dir, du_cache, tmp_path
):
    runner = CliRunner()

    struct_dir, _ = structure_dir
    du_cache_dir, _ = du_cache

    result = runner.invoke(
        cli,
        [
            "large-scale",
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
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_large_docking_cli_pdb_file(ligand_file, pdb_file, du_cache, tmp_path):
    runner = CliRunner()

    du_cache_dir, _ = du_cache

    result = runner.invoke(
        cli,
        [
            "large-scale",
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
        ],
    )
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


# TODO add tests for postera
