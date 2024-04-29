import argparse
import shlex
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
    "--input-type",
    type=str,
    required=False,
    default="fasta",
    help="Type of input between ['fasta', 'pdb', 'pre-calc']",
)

parser.add_argument(
    "--nalign",
    type=int,
    required=False,
    default=1000,
    help="Number of alignments that BLAST search will output",
)

parser.add_argument(
    "--e-thr",
    type=float,
    required=False,
    default=10,
    help="Threshold to select BLAST results",
)

parser.add_argument(
    "--sel-key",
    type=str,
    required=False,
    default="",
    help="Selection key to filter BLAST output. Provide either a keyword, or 'host: <species>'",
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

parser.add_argument(
    "--email",
    type=str,
    required=False,
    default="",
    help="Email for Entrez search",
)

parser.add_argument(
    "--multimer",
    default=False,
    action="store_true",
    help="Store the output sequences for a multimer ColabFold run (from identical chains)."
    'If not set, "--n-chains" will not be used. ',
)

parser.add_argument(
    "--n-chains",
    type=int,
    default=None,
    help="Number of repeated chains that will be saved in csv file."
    'Requires calling the "--multimer" option first.',
)

parser.add_argument(
    "--gen-ref-pdb",
    default=False,
    action="store_true",
    help="Whether to retrieve a pdb file for the query structure",
)


def main():
    args = parser.parse_args()
    # check all the required files exist
    fasta = Path(args.fasta)
    if not fasta.exists():
        raise FileNotFoundError(f"Fasta file {fasta} does not exist")
    if args.input_type in ["fasta", "pdb", "pre-calc"]:
        input_type = args.input_type
    else:
        raise ValueError(
            "The option input-type must be either 'fasta', 'pdb' or 'pre-calc'"
        )

    if "host" in args.sel_key:
        if len(args.email) > 0:
            email = args.email
        else:
            raise ValueError(
                "If a host selection is requested, an email must be provided"
            )

    n_chains = 1
    if args.multimer:
        n_chains = args.n_chains
    # Create folder if doesn't already exists
    results_folder = Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Perform BLAST search on input sequence
    matches_df = get_blast_seqs(
        args.fasta,
        results_folder,
        input_type=input_type,
        save_csv=args.save_blast,
        nalign=args.nalign,
        nhits=int(args.nalign * 3 / 4),
        e_val_thresh=args.e_thr,
        database="refseq_protein",
        verbose=False,
        email=email,
    )

    # Perform alignment for each entry in the FASTA file
    for query in matches_df["query"].unique():
        alignment = Alignment(matches_df, query, results_folder)
        file_prefix = f"{args.aln_output}{alignment.query_label}"
        selection_fasta, plot = do_MSA(
            alignment, args.sel_key, file_prefix, args.plot_width, n_chains
        )

        # Generate PDB file for template if requested (only for the reference structure)
        if args.gen_ref_pdb:
            pdb_entry = PDBEntry(seq=selection_fasta, type="fasta")
            pdb_file_record = pdb_entry.retrieve_pdb(
                results_folder=results_folder, min_id_match=99.9, ref_only=True
            )

            record = pdb_file_record[0]
            print(f"A PDB template for {record.label} was saved as {record.pdb_file}")

            # The following can be added to a shell file for running ColabFold
            file = open(results_folder / f"{file_prefix}_command.txt", "w")
            file.write("# Copy template PDB for ColabFold use\n")
            file.write(f'cp {shlex.quote(record.pdb_file)} "$template_path/0001.pdb"')
            file.close()

    return


if __name__ == "__main__":
    main()
