"""
Script to search the protein BLAST and return all matching sequences in a CSV
file that can then be passed to run_colabfold.sh.
"""

import argparse
from Bio.Blast import NCBIWWW, NCBIXML

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-f",
        "--fasta",
        required=True,
        help="Protein FASTA sequence to search in BLAST.",
    )
    parser.add_argument(
        "-o", "--out_fn", required=True, help="Output CSV file name."
    )
    parser.add_argument(
        "--cache", help="Optional cache file to save raw XML results."
    )

    parser.add_argument(
        "-aln",
        "--n_alignments",
        type=int,
        default=500,
        help="Number of alignments to retrieve.",
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

    # Load reference sequence
    fasta_seq = open(args.fasta).read().strip()

    # Run BLASTP and get results
    result_handle = NCBIWWW.qblast(
        program="blastp",
        database="refseq_protein",
        sequence=fasta_seq,
        alignments=args.n_alignments,
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
    with open(args.out_fn, "w") as fp:
        for k, v in result_seqs.items():
            fp.write(f">{k}\n{v}\n")


if __name__ == "__main__":
    main()
