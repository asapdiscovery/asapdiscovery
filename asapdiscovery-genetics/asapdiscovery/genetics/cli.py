from pathlib import Path
from typing import Optional

import click
import pandas as pd
from asapdiscovery.cli.cli_args import output_dir, pdb_file
from asapdiscovery.genetics.blast import PDBEntry, get_blast_seqs
from asapdiscovery.genetics.calculate_rmsd import (
    save_alignment_pymol,
    select_best_colabfold,
)
from asapdiscovery.genetics.cli_args import (
    blast_json,
    email,
    gen_ref_pdb,
    multimer,
    n_chains,
    seq_file,
    seq_type,
)
from asapdiscovery.genetics.seq_alignment import Alignment, do_MSA


@click.group()
def genetics():
    """Run genetics alignment workflows for related protein search and alignment."""
    pass


@genetics.command()
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
        selection_fasta, plot = do_MSA(
            alignment,
            sel_key,
            file_prefix,
            plot_width,
            n_chains,
            color_seq_match,
            align_start_idx,
        )

        # Generate PDB file for template if requested (only for the reference structure)
        if gen_ref_pdb:
            pdb_entry = PDBEntry(seq=selection_fasta, type="fasta")
            pdb_file_record = pdb_entry.retrieve_pdb(
                results_folder=results_folder, min_id_match=99.9, ref_only=True
            )

            record = pdb_file_record[0]
            print(f"A PDB template for {record.label} was saved as {record.pdb_file}")


@genetics.command()
@seq_file
@pdb_file
@output_dir
@click.option(
    "--cfold-results",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    default="./",
    help="Path to folder where all ColabFold results are stored.",
)
@click.option(
    "--pymol-save",
    type=str,
    default="aligned_proteins.pse",
    help="Path to save pymol session with aligned proteins.",
)
@click.option(
    "--cf-format",
    type=str,
    default="_unrelaxed_rank_001_alphafold2_ptm",
    help="Format of pdb file saved by ColabFold, according to the folding model and relaxation used.",
)
def struct_alignment(
    seq_file: str,
    pdb_file: str,
    cfold_results: Optional[str] = "./",
    pymol_save: Optional[str] = "aligned_proteins.pse",
    cf_format: Optional[str] = "_unrelaxed_rank_001_alphafold2_ptm",
    output_dir: str = "output",
):
    """
    Align PDB structures generated from ColabFold with respect to a reference pdb_file, as listed in the csv seq_file used for the folding.
    """

    if not Path(seq_file).exists():
        raise FileNotFoundError(f"Sequence file {seq_file} does not exist")

    ref_pdb = Path(pdb_file)
    if not ref_pdb.exists():
        raise FileNotFoundError(f"Ref PDB file {ref_pdb} does not exist")

    results_dir = Path(cfold_results)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"The folder with ColabFold results {results_dir} does not exist"
        )
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    aligned_pdbs = []
    seq_labels = []
    seq_df = pd.read_csv(seq_file)
    for index, row in seq_df.iterrows():
        # iterate over each csv entry
        mol = row["id"]
        final_pdb = save_dir / f"{mol}_aligned.pdb"
        # Select best seed repetition
        min_rmsd, min_file = select_best_colabfold(
            results_dir,
            mol,
            ref_pdb,
            chain="A",
            final_pdb=final_pdb,
            fold_model=cf_format,
        )

        aligned_pdbs.append(min_file)
        seq_labels.append(mol)

    session_save = save_dir / pymol_save
    save_alignment_pymol(aligned_pdbs, seq_labels, ref_pdb, session_save)


if __name__ == "__main__":
    genetics()
