from pathlib import Path
from typing import Optional

import click
import pandas as pd
from asapdiscovery.cli.cli_args import output_dir, pdb_file
from asapdiscovery.spectrum.align_seq_match import (
    fasta_alignment,
    pairwise_alignment,
    save_pymol_seq_align,
)
from asapdiscovery.spectrum.blast import PDBEntry, get_blast_seqs
from asapdiscovery.spectrum.calculate_rmsd import (
    save_alignment_pymol,
    select_best_colabfold,
)
from asapdiscovery.spectrum.cli_args import (
    blast_json,
    email,
    gen_ref_pdb,
    max_mismatches,
    multimer,
    n_chains,
    pymol_save,
    seq_file,
    seq_type,
)
from asapdiscovery.spectrum.seq_alignment import Alignment, do_MSA


@click.group()
def spectrum():
    """Run spectrum alignment workflows for related protein search and alignment."""
    pass


@spectrum.command()
@seq_file
@seq_type
@output_dir
@click.option(
    "--nalign",
    type=int,
    default=1000,
    help="Number of alignments that BLAST search will output.",
)
@click.option(
    "--e-thr",
    type=float,
    default=10.0,
    help="Threshold to select BLAST results.",
)
@click.option(
    "--save-blast",
    type=str,
    default="blast.csv",
    help="Optional file name for saving result of BLAST search",
)
@click.option(
    "--sel-key",
    type=str,
    default="",
    help="Selection key to filter BLAST output. Provide either a keyword, or 'host: <species>'",
)
@blast_json
@email
@multimer
@n_chains
@gen_ref_pdb
@click.option(
    "--plot-width",
    type=int,
    default=1500,
    help="Width for the multi-alignment plot.",
)
@click.option(
    "--color-seq-match",
    is_flag=True,
    default=False,
    help="Color aminoacid matches in html alignment: Red for exact match and yellow for same-group match.",
)
@click.option(
    "--align-start-idx",
    default=0,
    help="Start index for reference aminoacids in html alignment (Useful when matching idxs to PyMOL labels)",
)
@max_mismatches
@click.option(
    "--custom-order",
    default="",
    help="Custom order of aligned sequences (not including ref) can be provided as a string with comma-sep indexes.",
)
def seq_alignment(
    seq_file: str,
    seq_type: Optional[str] = None,
    nalign: int = 1000,
    e_thr: float = 10.0,
    sel_key: str = "",
    plot_width: int = 1500,
    blast_json: Optional[str] = None,
    save_blast: Optional[str] = "blast.csv",
    email: str = "",
    multimer: bool = False,
    n_chains: int = 1,
    gen_ref_pdb: bool = False,
    output_dir: str = "output",
    color_seq_match: bool = False,
    align_start_idx: int = 0,
    max_mismatches: int = 2,
    custom_order: str = "",
):
    """
    Find similarities between reference protein and its related proteins by sequence.
    """

    if blast_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        raise NotImplementedError("Haven't implement the json option yet")
    else:
        pass

    # check all the required files exist
    if not Path(seq_file).exists():
        raise FileNotFoundError(f"Fasta file {seq_file} does not exist")
    if seq_type in ["fasta", "pdb", "pre-calc"]:
        input_type = seq_type
    else:
        raise ValueError(
            "The option input-type must be either 'fasta', 'pdb' or 'pre-calc'"
        )

    if multimer:
        n_chains = n_chains
    else:
        n_chains = 1
    # Create folder if doesn't already exists
    results_folder = Path(output_dir)
    results_folder.mkdir(parents=True, exist_ok=True)

    if "host" in sel_key:
        if len(email) < 0:
            raise ValueError(
                "If a host selection is requested, an email must be provided"
            )

    # Perform BLAST search on input sequence
    matches_df = get_blast_seqs(
        seq_file,
        results_folder,
        input_type=input_type,
        save_csv=save_blast,
        nalign=nalign,
        nhits=int(nalign * 3 / 4),
        e_val_thresh=e_thr,
        database="refseq_protein",
        verbose=False,
        email=email,
    )

    # Perform alignment for each entry in the FASTA file
    for query in matches_df["query"].unique():
        alignment = Alignment(matches_df, query, results_folder)
        file_prefix = alignment.query_label
        alignment_out = do_MSA(
            alignment,
            sel_key,
            file_prefix,
            plot_width,
            n_chains,
            color_seq_match,
            align_start_idx,
            max_mismatches,
            custom_order,
        )

        # Generate PDB file for template if requested (only for the reference structure)
        if gen_ref_pdb:
            pdb_entry = PDBEntry(seq=alignment_out.select_file, type="fasta")
            pdb_file_record = pdb_entry.retrieve_pdb(
                results_folder=results_folder, min_id_match=99.9, ref_only=True
            )

            record = pdb_file_record[0]
            print(f"A PDB template for {record.label} was saved as {record.pdb_file}")


@spectrum.command()
@seq_file
@pdb_file
@output_dir
@click.option(
    "--cfold-results",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to folder where all ColabFold results are stored.",
)
@click.option(
    "--pdb-align",
    type=str,
    help="Path to PDB to align. Not needed when --cfold-results is given.",
)
@click.option(
    "--struct-dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to folder where structures to align are stored. Not needed when --cfold-results or --pdb-align is given.",
)
@click.option(
    "--pymol-save",
    type=str,
    default="aligned_proteins.pse",
    help="Path to save pymol session with aligned proteins.",
)
@click.option(
    "--chain",
    type=str,
    default="both",
    help="Chains to display on visualization ('A', 'B' or 'both'). The default 'both' will align wrt chain A but display both chains.",
)
@click.option(
    "--color-by-rmsd",
    is_flag=True,
    default=False,
    help="Option to generate a PyMOL session were targets are colored by RMSD with respect to ref.",
)
@click.option(
    "--cf-format",
    type=str,
    default="alphafold2_ptm",
    help="Model used with ColabFold. Either 'alphafold2_ptm' or 'alphafold2_multimer_v3'",
)
def struct_alignment(
    seq_file: str,
    pdb_file: str,
    cfold_results: Optional[str] = None,
    struct_dir: Optional[str] = None,
    pdb_align: Optional[str] = None,
    pymol_save: Optional[str] = "aligned_proteins.pse",
    color_by_rmsd: Optional[bool] = False,
    chain: Optional[str] = "A",
    cf_format: Optional[str] = "alphafold2_ptm",
    output_dir: str = "output",
):
    """
    Align PDB structures generated from ColabFold with respect to a reference pdb_file, as listed in the csv seq_file used for the folding.
    """

    ref_pdb = Path(pdb_file)
    if not ref_pdb.exists():
        raise FileNotFoundError(f"Ref PDB file {ref_pdb} does not exist")

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    session_save = save_dir / pymol_save

    if not (cfold_results or struct_dir or pdb_align):
        raise ValueError(
            "At least one of 'cfold_results', 'struct_dir', or 'pdb_align' must be provided."
        )

    if cfold_results is None:  # cfold results has priority
        session_save = save_dir / pymol_save
        if pdb_align is not None:  # priority given to pdb_align
            results_dir = Path(pdb_align)
            aligned_pdbs = [str(results_dir)]
            seq_labels = [results_dir.stem]
        else:
            results_dir = Path(struct_dir)
            aligned_pdbs = []
            seq_labels = []
            for file_path in results_dir.glob("*.pdb"):
                print(f"Reading structure {file_path.stem}")
                aligned_pdbs.append(str(file_path))
                seq_labels.append(file_path.stem)
        if not results_dir.exists():
            raise FileNotFoundError(
                f"The folder with pdbs to align {results_dir} does not exist"
            )
        save_alignment_pymol(
            aligned_pdbs, seq_labels, ref_pdb, session_save, chain, color_by_rmsd
        )
        return
    else:
        # ColabFold results pipeline
        results_dir = Path(cfold_results)
        if not results_dir.exists():
            raise FileNotFoundError(
                f"The folder with ColabFold results {results_dir} does not exist"
            )
        if not Path(seq_file).exists():
            raise FileNotFoundError(f"Sequence file {seq_file} does not exist")
        aligned_pdbs = []
        seq_labels = []
        seq_df = pd.read_csv(seq_file)
        seq_df.columns = seq_df.columns.str.lower()  # make case-insensitive
        for index, row in seq_df.iterrows():
            # iterate over each csv entry
            mol = row["id"]
            final_pdb = save_dir / f"{mol}_aligned.pdb"
            # Select best seed repetition
            align_chain = chain
            if chain == "both":
                align_chain = "A"
            min_rmsd, min_file = select_best_colabfold(
                results_dir,
                mol,
                ref_pdb,
                chain=align_chain,
                final_pdb=final_pdb,
                fold_model=cf_format,
            )
            aligned_pdbs.append(min_file)
            seq_labels.append(mol)

    session_save = save_dir / pymol_save
    save_alignment_pymol(
        aligned_pdbs, seq_labels, ref_pdb, session_save, chain, color_by_rmsd
    )


@spectrum.command()
@pdb_file
@click.option(
    "-t",
    "--type",
    type=str,
    default="pwise",
    help="If 'pwise', a pairwise alignment is done with pdb-complex. With 'fasta', a fasta file is provided with the precomputed alignment.",
)
@click.option(
    "--pdb-align",
    type=str,
    help="Path to PDB to align",
)
@click.option(
    "--struct-dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to folder where structures to align are stored. Not needed when --pdb-align is given.",
)
@click.option(
    "--pdb-label",
    type=str,
    default="ref,pdb",
    help="Label of PDB in PyMOL (optional). Provide as string 'REF,ALIGN1,<ALIGN2>,<ALIGN3>,...'",
)
@pymol_save
@click.option(
    "--fasta-a",
    type=str,
    default=None,
    help="Path to fasta with chain A alignment",
)
@click.option(
    "--fasta-b",
    type=str,
    default=None,
    help="Path to fasta with chain B alignment",
)
@click.option(
    "--fasta-sel",
    type=str,
    default="0,1",
    help="Index of sequences in fasta file to use in the alignment (optional for --struct-dir mode).",
)
@click.option(
    "--start-a",
    type=int,
    default="1",
    help="Start index for chain A. In multi-sequence alignment mode, all proteins to align must have the same start idx",
)
@click.option(
    "--start-b",
    type=int,
    default="1",
    help="Start index for chain B. In multi-sequence alignment mode, all proteins to align must have the same start idx",
)
@max_mismatches
def fitness_alignment(
    pdb_file: str,
    pdb_label: str,
    type: str,
    pymol_save: str,
    pdb_align: str,
    struct_dir: str,
    fasta_sel: str,
    start_a=1,
    start_b=1,
    fasta_a=None,
    fasta_b=None,
    max_mismatches=0,
) -> None:
    """
    Align PDB structures and color by parwise or multi-sequence alignment match
    """

    start_idxA = start_a
    start_idxB = start_b

    session_save = pymol_save
    pdb_labels = pdb_label.split(",")
    if type == "pwise":
        if pdb_align is None:
            raise ValueError(
                "pdb-align must be provided in pairwise mode! struct-dir pairwise alignment is not possible."
            )
        pdb_align, colorsA, colorsB = pairwise_alignment(
            pdb_file, pdb_align, start_idxA, start_idxB
        )
    elif type == "fasta":
        assert fasta_a is not None
        assert fasta_b is not None
        pdb_align, colorsA, colorsB, pdb_labels = fasta_alignment(
            fasta_a,
            fasta_b,
            fasta_sel,
            pdb_labels,
            start_idxA,
            start_idxB,
            pdb_align,
            struct_dir,
            max_mismatches,
        )
    else:
        raise NotImplementedError("Types allowed are 'pwise' and 'fasta'")
    save_pymol_seq_align(
        pdb_align, pdb_labels, pdb_file, [colorsA, colorsB], session_save
    )


if __name__ == "__main__":
    spectrum()
