import os
import traceback

import pandas as pd
import pytest
from asapdiscovery.genetics.blast import pdb_to_seq
from asapdiscovery.genetics.cli import genetics as cli
from asapdiscovery.genetics.seq_alignment import Alignment, do_MSA
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_pdb_to_seq_no_out(protein_path):
    from Bio import SeqRecord

    seq_record = pdb_to_seq(
        pdb_input=protein_path,
        chain="A",
        fasta_out=None,
    )
    assert isinstance(seq_record, SeqRecord.SeqRecord)
    assert len(seq_record.seq) > 0


def test_pdb_to_seq(protein_path, tmp_path):
    seq_record, fout = pdb_to_seq(
        pdb_input=protein_path,
        chain="A",
        fasta_out=tmp_path / "test.fasta",
    )


# @pytest.mark.skipif(
#     os.getenv("RUNNER_OS") == "macOS",
#     reason="Test failing on GHA runner, fine locally.",
# )
def test_MSA_host_key(blast_csv_path, tmp_path):
    blast_csv = pd.read_csv(blast_csv_path)
    alignment = Alignment(blast_csv, blast_csv["query"][0], tmp_path)
    aln_out = do_MSA(
        alignment=alignment,
        select_mode="host: Homo sapiens OR organism: human",
        file_prefix=alignment.query_label,
        plot_width=1000,
        n_chains=1,
        color_by_group=False,
        start_alignment_idx=0,
        max_mismatch=2,
    )
    assert aln_out.sucess
    assert all("Homo sapiens" in a or "Not found" in a for a in aln_out.hosts)
    assert len(aln_out.align_obj) > 1
    assert all(len(a) == len(aln_out.align_obj[0]) for a in aln_out.align_obj)


# @pytest.mark.skipif(
#     os.getenv("RUNNER_OS") == "macOS",
#     reason="Test failing on GHA runner, fine locally.",
# )
def test_MSA_keyword(blast_csv_path, tmp_path):
    blast_csv = pd.read_csv(blast_csv_path)
    alignment = Alignment(blast_csv, blast_csv["query"][0], tmp_path)
    aln_out = do_MSA(
        alignment=alignment,
        select_mode="Middle East respiratory syndrome-related coronavirus,Human coronavirus HKU1",
        file_prefix=alignment.query_label,
        plot_width=1000,
        n_chains=1,
        color_by_group=False,
        start_alignment_idx=0,
        max_mismatch=2,
    )
    assert aln_out.sucess
    assert len(aln_out.align_obj) == 3
    assert all(len(a) == len(aln_out.align_obj[0]) for a in aln_out.align_obj)


# @pytest.mark.skipif(
#     os.getenv("RUNNER_OS") == "macOS",
#     reason="Test failing on GHA runner, fine locally.",
# )
def test_MSA_color_match(blast_csv_path, tmp_path):
    blast_csv = pd.read_csv(blast_csv_path)
    alignment = Alignment(blast_csv, blast_csv["query"][0], tmp_path)
    aln_out = do_MSA(
        alignment=alignment,
        select_mode="",
        file_prefix=alignment.query_label,
        plot_width=1000,
        n_chains=1,
        color_by_group=True,
        start_alignment_idx=0,
        max_mismatch=2,
    )
    assert aln_out.sucess


@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow on macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_seq_alignment_pre_calc(blast_xml_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "seq-alignment",
            "-f",
            blast_xml_path,
            "-t",
            "pre-calc",
            "--sel-key",
            "",
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)


@pytest.mark.skipif(os.getenv("RUNNER_OS") == "macOS", reason="Slow in macOS")
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_seq_alignment_multimer(blast_xml_path, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "seq-alignment",
            "-f",
            blast_xml_path,
            "-t",
            "pre-calc",
            "--sel-key",
            "",
            "--multimer",
            "--n-chains",
            2,
            "--output-dir",
            tmp_path,
        ],
    )
    assert click_success(result)
