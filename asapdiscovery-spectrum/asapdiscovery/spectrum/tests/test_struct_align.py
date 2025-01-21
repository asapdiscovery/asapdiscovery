import os
import traceback

import pytest
from asapdiscovery.spectrum.cli import spectrum as cli
from asapdiscovery.spectrum.calculate_rmsd import rmsd_alignment, save_alignment_pymol
from asapdiscovery.spectrum.align_seq_match import pairwise_alignment, save_pymol_seq_align
from click.testing import CliRunner

def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_rmsd_alignment(protein_path, protein_apo_path, tmp_path):
    rmsd, pdb_out = rmsd_alignment(
        target_pdb=protein_apo_path,
        ref_pdb=protein_path,
        final_pdb=tmp_path / "file.pdb",
        target_chain="A",
        ref_chain="A",
    )
    assert isinstance(rmsd, float)
    assert pdb_out.exists()

def test_save_alignment(protein_path, protein_apo_path, tmp_path):
    pse_out = tmp_path / "file.pse"
    save_alignment_pymol(
        pdbs=[protein_apo_path],
        labels=["pdb"],
        reference=protein_path,
        session_save=pse_out,
        align_chain="A",
    )
    assert pse_out.exists()


def test_pairwise_alignment(protein_path):
    # Test of pairwise alignment with the same protein file twice
    start_idx = 1
    pdb_align, colorsA, colorsB = pairwise_alignment(
        pdb_file=protein_path,
        pdb_align=protein_path,
        start_idxA=start_idx,
        start_idxB=start_idx,
    )
    assert len(pdb_align)==1
    assert len(set(colorsA.values()))==1 # All should be white
    assert len(set(colorsB.values()))==1
    assert colorsA[start_idx] == "white"
    assert colorsB[start_idx] == "white"

def test_pymol_seq_align(protein_path, tmp_path):
    import MDAnalysis as mda
    u = mda.Universe(protein_path)
    nres = len(u.select_atoms("protein").residues)
    colorsA = {
        (index + 1): string for index, string in enumerate(["white"]*nres)
    }
    pse_out = tmp_path / "file.pse"

    save_pymol_seq_align(
        pdbs=[protein_path],
        labels=["ref","pdb"],
        reference=protein_path,
        color_dict=[colorsA, colorsA],
        session_save=pse_out,
    )
    assert pse_out.exists()

@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow on macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_single_pdb(blast_csv_path, protein_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_path,
            "--pymol-save",
            tmp_path/"file.pse",
            "--chain",
            "both",
            "--color-by-rmsd"
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow in macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_one_chain(blast_csv_path, protein_path, protein_apo_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_apo_path,
            "--pymol-save",
            tmp_path/"file.pse",
            "--chain",
            "A",
            "--color-by-rmsd"
        ],
    )
    assert click_success(result)

@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow in macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_struct_dir(blast_csv_path, protein_path, structure_dir, tmp_path):
    runner = CliRunner()
    struct_dir, _ = structure_dir
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--struct-dir",
            struct_dir,
            "--pymol-save",
            tmp_path/"file.pse",
            "--chain",
            "both",
            "--color-by-rmsd"
        ],
    )
    assert click_success(result)

@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow in macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_struct_alignment_cfold_dir(blast_csv_path, protein_path, cfold_dir, tmp_path):
    runner = CliRunner()
    struct_dir, _ = structure_dir
    result = runner.invoke(
        cli,
        [
            "struct-alignment",
            "-f",
            blast_csv_path,
            "--pdb-file",
            protein_path,
            "--cfold-results",
            cfold_dir,
            "--pymol-save",
            tmp_path/"file.pse",
            "--chain",
            "both",
            "--color-by-rmsd",
            "--cf-format",
            "alphafold2_multimer_v3",
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow in macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_fitness_alignment_pairwise(blast_csv_path, protein_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "fitness-alignment",
            "-t",
            "pwise",
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_path,
            "--pdb-label",
            "ref,pdb",
            "--pymol-save",
            tmp_path/"file.pse",
        ],
    )
    assert click_success(result)

@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow in macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_fitness_alignment_fasta(blast_csv_path, fasta_alignment_path, protein_path, protein_mers_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "fitness-alignment",
            "-t",
            "fasta",
            "--pdb-file",
            protein_path,
            "--pdb-align",
            protein_mers_path,
            "--pdb-label",
            "ref,pdb",
            "--pymol-save",
            tmp_path/"file.pse",
            "--fasta-a",
            fasta_alignment_path,
            "--fasta-b",
            fasta_alignment_path,
            "--fasta-sel",
            "0,4",
        ],
    )
    assert click_success(result)

