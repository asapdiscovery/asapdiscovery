"""
Script to search the protein BLAST and return all matching sequences in a CSV
file that can then be passed to run_colabfold.sh.
"""

import argparse
from Bio.Blast import NCBIWWW, NCBIXML
import os

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-if",
        "--in_fasta",
        required=True,
        help="Protein FASTA sequence to search in BLAST.",
    )
    parser.add_argument(
        "-of",
        "--out_fasta",
        help="FASTA file for BLAST results.",
    )
    parser.add_argument(
        "-o", "--out_fn", required=True, help="Output CSV file name."
    )
    parser.add_argument(
        "--cache", help="Optional cache file to save raw XML results."
    )

    parser.add_argument(
        "-n",
        "--n_hits",
        type=int,
        default=500,
        help="Number of hits to return.",
    )
    parser.add_argument(
        "-e",
        "--e_val_thresh",
        type=float,
        default=1e-20,
        help="Threshold for BLAST E value.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Check for cache and load if present
    if args.cache and os.path.isfile(args.cache):
        result_handle = open(args.cache)
    else:
        # Load reference sequence
        fasta_seq = open(args.in_fasta).read().strip()

        # Run BLASTP and get results
        result_handle = NCBIWWW.qblast(
            program="blastp",
            database="refseq_protein",
            sequence=fasta_seq,
            hitlist_size=args.n_hits,
        )

        # Save BLAST results
        if args.cache:
            with open(args.cache, "w") as fp:
                fp.write(result_handle.read())
            result_handle.seek(0)

    # Parse BLAST results
    result_seqs = {}
    for record in NCBIXML.parse(result_handle):
        if record.alignments:
            for align in record.alignments:
                for hsp in align.hsps:
                    if hsp.expect < args.e_val_thresh:
                        # Save sequence identity, title, and gapless sequence
                        #  substring that aligns
                        sequence_to_model = align.hsps[0].sbjct.replace("-", "")
                        result_seqs[align.title] = sequence_to_model
    print(f"Found {len(result_seqs)} sequences", flush=True)

    # Write FASTA file
    if args.out_fasta:
        with open(args.out_fasta, "w") as fp:
            for k, v in result_seqs.items():
                fp.write(f">{k}\n{v}\n")

    # Write CSV file
    with open(args.out_fn, "w") as fp:
        fp.write("id,sequence\n")
        for k, v in result_seqs.items():
            accession = k.split("|")[1]
            # Need to write out the sequence twice for dimer
            fp.write(f"{accession},{v}:{v}\n")


if __name__ == "__main__":
    main()
