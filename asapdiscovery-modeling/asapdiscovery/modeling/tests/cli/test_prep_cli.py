import os
import pathlib
import traceback

import numpy as np
import pytest
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
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
def test_prep_cli_pdb_file(pdb_file, tmpdir):
    """Test preparing from a pdb file and that the structure is aligned to the reference structure."""
    runner = CliRunner()

    # load the input pdb and not the atom positions of the ligand
    input_complex = Complex.from_pdb(
        pdb_file, target_kwargs={"target_name": "align_test"}
    )
    oe_ligand = input_complex.ligand.to_oemol()
    ligand_coords = np.array(list(oe_ligand.GetCoords().values())).reshape((-1, 3))
    with tmpdir.as_cwd():
        result = runner.invoke(
            cli,
            [
                "protein-prep",
                "--target",
                "SARS-CoV-2-Mpro",
                "--pdb-file",
                pdb_file,
                "--output-dir",
                "output",
            ],
        )
        assert click_success(result)
        # load the final complex and check the heavy atom positions of the ligand
        prepped_complex = PreppedComplex.from_json_file(
            pathlib.Path("output").joinpath(
                "Mpro-P2660_0A_bound_oe_processed-c9e7ff3683441e1c1848ea0a6a699901d81135e2cde78e1b6ff0e160e4f06f2a+JZJCSVMJFIAMQB-DLYUOGNHNA-N",
                "Mpro-P2660_0A_bound_oe_processed.json",
            )
        )
        final_ligand_coords = np.array(
            list(prepped_complex.ligand.to_oemol().GetCoords().values())
        ).reshape((-1, 3))[: oe_ligand.NumAtoms()]
        assert not np.allclose(ligand_coords, final_ligand_coords)


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


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Prep tests slow on GHA on macOS"
)
def test_prep_cli_cache_reuse(structure_dir, json_cache, tmp_path):
    """Make sure cached structures are reused when running the cli"""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "protein-prep",
            "--target",
            "SARS-CoV-2-Mpro",
            "--structure-dir",
            structure_dir[0],
            "--cache-dir",
            json_cache.parent,
            "--no-save-to-cache",
            "--output-dir",
            str(tmp_path.joinpath("output")),
        ],
    )

    assert click_success(result)
    assert "Loaded 2 complexes" in result.output
    assert "Loaded 1 cached structures" in result.output
    assert "Matched 1 cached structures" in result.output
    assert "Prepping 1 complexes" in result.output
    assert "Prepped 2 complexes" in result.output
