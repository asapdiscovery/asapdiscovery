"""
Script to search the protein BLAST and return all matching sequences in a CSV
file that can then be passed to run_colabfold.sh.
"""

import argparse
import os
import re

from Bio.Blast import NCBIWWW, NCBIXML


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
    parser.add_argument("-o", "--out_fn", required=True, help="Output CSV file name.")
    parser.add_argument("--cache", help="Optional cache file to save raw XML results.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cache file if it already exists.",
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
    parser.add_argument(
        "-f",
        "--filter",
        nargs="+",
        default=[],
        help="Regex(es) to search in hit title.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Check for cache and load if present
    if args.cache and os.path.isfile(args.cache) and (not args.overwrite):
        print("Loading from cached download", flush=True)
        result_handle = open(args.cache)
    else:
        print("Searching BLASTP", flush=True)
        # Load reference sequence
        fasta_seq = open(args.in_fasta).read().strip()

        # Run BLASTP and get results
        result_handle = NCBIWWW.qblast(
            program="blastp",
            database="refseq_protein",
            sequence=fasta_seq,
            hitlist_size=args.n_hits,
            expect=args.e_val_thresh,
        )

        # Save BLAST results
        if args.cache:
            print("Saving cache", flush=True)
            with open(args.cache, "w") as fp:
                fp.write(result_handle.read())
            result_handle.seek(0)

    # Parse BLAST results
    result_seqs = {}
    total_results = 0
    for record in NCBIXML.parse(result_handle):
        if not record.alignments:
            continue

        for align in record.alignments:
            total_results += 1
            if (len(args.filter) == 0) or any(
                re.search(f, align.title) for f in args.filter
            ):
                # Save sequence identity, title, and gapless sequence
                #  substring that aligns
                sequence_to_model = align.hsps[0].sbjct.replace("-", "")
                result_seqs[align.title] = sequence_to_model
    print(f"Kept {len(result_seqs)} out of {total_results} hits", flush=True)

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
            fp.write(f"{accession}_{{}},{v}:{v}\n")


if __name__ == "__main__":
    main()
