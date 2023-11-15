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


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("subcommand, posit_cutoff", [("large-scale", 0), ("small-scale", 0), ("cross-docking", False)])
@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking_cli_fragalysis(
    ligand_file, mpro_frag_dir, tmp_path, subcommand, use_dask, posit_cutoff
):
    runner = CliRunner()

    frag_parent_dir, _ = mpro_frag_dir
    cli_args = [
        subcommand,
        "--target",
        "SARS-CoV-2-Mpro",
        "--ligands",
        ligand_file,
        "--fragalysis-dir",
        frag_parent_dir,
        use_dask,
        "--output-dir",
        tmp_path,
    ]

    if use_dask:
        cli_args += ["--use-dask"]

    if posit_cutoff:
        cli_args += ["--posit-confidence-cutoff", posit_cutoff]

    result = runner.invoke(cli, cli_args)
    assert click_success(result)


@pytest.mark.parametrize("subcommand, posit_cutoff", [("large-scale", 0), ("small-scale", 0), ("cross-docking", False)])
@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking_cli_structure_directory_du_cache(
    ligand_file, structure_dir, du_cache, tmp_path, subcommand, posit_cutoff
):
    runner = CliRunner()

    struct_dir, _ = structure_dir
    du_cache_dir, _ = du_cache

    cli_args =  [
            subcommand,
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
        ]
    
    if posit_cutoff:
        cli_args += ["--posit-confidence-cutoff", posit_cutoff]
    

    result = runner.invoke(
        cli,
        cli_args,
       
    )
    assert click_success(result)


@pytest.mark.parametrize("subcommand, posit_cutoff", [("large-scale", 0), ("small-scale", 0), ("cross-docking", False)])
@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
def test_docking_cli_pdb_file(ligand_file, pdb_file, tmp_path, subcommand, posit_cutoff):
    runner = CliRunner()

    cli_args =  [
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

    
    if posit_cutoff:
        cli_args += ["--posit-confidence-cutoff", posit_cutoff]
    

    result = runner.invoke(
        cli,
        cli_args,

    )
    assert click_success(result)
