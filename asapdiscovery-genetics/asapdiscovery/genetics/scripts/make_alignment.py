import argparse
import logging
from pathlib import Path

from asapdiscovery.genetics.blast import PDBEntry, get_blast_seqs
from asapdiscovery.genetics.seq_alignment import Alignment, do_MSA

parser = argparse.ArgumentParser(
    description="Find similarities between reference protein and its related proteins by sequence"
)

parser.add_argument(
    "--fasta",
    type=str,
    required=True,
    help="Path to input fasta file with ref sequence",
)
parser.add_argument(
    "--results-folder",
    type=str,
    required=True,
    help="Path to folder for storing results",
)

parser.add_argument(
    "--nalign",
    type=int,
    required=False,
    default=1000,
    help="Number of alignments that BLAST search will output",
)

parser.add_argument(
    "--sel-key",
    type=str,
    required=False,
    default="human",
    help="Selection key to filter BLAST output",
)

parser.add_argument(
    "--aln-output",
    type=str,
    required=False,
    default="",
    help="Optional prefix for output files",
)

parser.add_argument(
    "--plot-width",
    type=int,
    required=False,
    default=1500,
    help="Width of multi-alignment plot",
)

parser.add_argument(
    "--save-blast",
    type=str,
    required=False,
    default="blast.csv",
    help="Optional file name for saving result of BLAST search",
)


def main():
    args = parser.parse_args()
    # check all the required files exist
    fasta = Path(args.fasta)
    if not fasta.exists():
        raise FileNotFoundError(f"Pose file {fasta} does not exist")
    # Create folder if doesn't already exists
    results_folder = Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    # Perform BLAST search on input sequence
    matches_df = get_blast_seqs(
        args.fasta,
        results_folder,
        input_type="fasta",
        save_csv=args.save_blast,
        nalign=args.nalign,
        nhits=args.nalign * 3 / 4,
        database="refseq_protein",
        verbose=False,
    )

    # Perform alignment for each entry in the FASTA file
    for query in matches_df["query"].unique():
        alignment = Alignment(matches_df, query, results_folder)
        file_prefix = f"{args.aln_output}{alignment.query_label}"
        selection_fasta, plot = do_MSA(
            alignment, args.sel_key, file_prefix, args.plot_width
        )

        # Generate PDB file for template (only for the reference structure)
        pdb_entry = PDBEntry(seq=selection_fasta, type="fasta")
        pdb_file_record = pdb_entry.retrieve_pdb(
            results_folder=results_folder, min_id_match=99.9, ref_only=True
        )

        record = pdb_file_record[0]
        print(f"A PDB template for {record.label} was saved as {record.pdb_file}")

    return


if __name__ == "__main__":
    main()
