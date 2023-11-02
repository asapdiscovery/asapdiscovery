import os
import traceback

import pytest
from asapdiscovery.modeling.cli import modeling as cli
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Prep tests slow on GHA on macOS"
)
def test_prep_cli_fragalysis(mpro_frag_dir, tmp_path):
    runner = CliRunner()

    frag_parent_dir, _ = mpro_frag_dir

    result = runner.invoke(
        cli,
        [
            "protein-prep",
            "--target",
            "SARS-CoV-2-Mpro",
            "--fragalysis-dir",
            frag_parent_dir,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Prep tests slow on GHA on macOS"
)
def test_prep_cli_structure_dir(structure_dir, tmp_path):
    runner = CliRunner()

    structure_directory, _ = structure_dir

    result = runner.invoke(
        cli,
        [
            "protein-prep",
            "--target",
            "SARS-CoV-2-Mpro",
            "--structure-dir",
            structure_directory,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Prep tests slow on GHA on macOS"
)
def test_prep_cli_pdb_file(pdb_file, tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "protein-prep",
            "--target",
            "SARS-CoV-2-Mpro",
            "--pdb-file",
            pdb_file,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Prep tests slow on GHA on macOS"
)
def test_prep_cli_pdb_file_align(pdb_file, tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "protein-prep",
            "--target",
            "SARS-CoV-2-Mpro",
            "--pdb-file",
            pdb_file,
            "--align",
            pdb_file,
            "--ref-chain",
            "A",
            "--active-site-chain",
            "A",
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)
