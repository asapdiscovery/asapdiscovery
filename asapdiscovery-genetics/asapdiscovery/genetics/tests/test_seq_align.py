import os
import traceback
from unittest import mock

import pytest
from asapdiscovery.genetics.blast import pdb_to_seq 
from asapdiscovery.genetics.seq_alignment import Alignment

from click.testing import CliRunner
from openmm import unit


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_pdb_to_seq_no_out(protein_path):
    from Bio import SeqRecord
    with pytest.raises(ValueError):
        seq_record = pdb_to_seq(
                pdb_input=protein_path,
                chain="A", 
                fasta_out=None,
        )
    assert type(seq_record) == SeqRecord.SeqRecord
    assert len(seq_record.seq) > 0

def test_pdb_to_seq(protein_path, tmp_path):
    with pytest.raises(ValueError):
        seq_record, fout = pdb_to_seq(
                pdb_input=protein_path, 
                chain="A",
                fasta_out=temp_path,
        )

def test_MSA_host_key(alignment):
    aln_out = do_MSA(
            alignment=alignment,
            select_mode="host: Homo sapiens OR organism: human",
            file_prefix=alignment.query_label,
            plot_width=1000,
            n_chains=1,
            color_by_group=False,
            start_alignment_idx=0,
    )
    assert aln_out.sucess
    assert all('Homo sapiens' in a or 'Not found' in a for a in aln_out.hosts)
    assert len(aln_out.align_obj) > 1
    assert all(len(a)==len(aln_out.align_obj[0]) for a in aln_out.align_obj)

def test_MSA_keyword(alignment):
    do_MSA(
        alignment=alignment,
        select_mode="Middle East respiratory syndrome-related coronavirus,Human coronavirus HKU1",
        file_prefix=alignment.query_label,
        plot_width=1000,
        n_chains=1,
        color_by_group=False,
        start_alignment_idx=0,
    )
    assert aln_out.sucess
    assert len(aln_out.align_obj) == 3
    assert all(len(a)==len(aln_out.align_obj[0]) for a in aln_out.align_obj)

def test_MSA_color_match(alignment):
    do_MSA(
        alignment=alignment,
        select_mode="",
        file_prefix=alignment.query_label,
        plot_width=1000,
        n_chains=1,
        color_by_group=True,
        start_alignment_idx=0,
    )
    assert aln_out.sucess

@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_seq_alignment_pre_calc(
    blast_xml_path, tmp_path
):
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
    ]
    result = runner.invoke(cli, args)
    assert click_success(result)

@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_seq_alignment_multimer(
    blast_xml_path, tmp_path
):
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
    ]
    result = runner.invoke(cli, args)
    assert click_success(result)

