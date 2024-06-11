import argparse
from pathlib import Path

from Bio import SeqIO
from Bio.SeqUtils import seq3


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--in_file", required=True, type=Path, help="Input FASTA file."
    )
    parser.add_argument(
        "-o", "--out_file", required=True, type=Path, help="Output YAML file."
    )

    return parser.parse_args()


def main():
    args = get_args()

    seqres_str = "SEQRES: |\n"
    for r in SeqIO.parse(args.in_file.open("r"), "fasta"):
        # Based on the format Andre used in Slack
        rec_chains = r.description.split("|")[1]
        rec_chains = [c.strip(",") for c in rec_chains.split()[1:]]

        # Number of residues in the chain(s)
        rec_len = len(r.seq)

        # Conver to 3-letter codes and split up
        rec_seq = seq3(r.seq)
        rec_residues = [rec_seq[i : i + 3].upper() for i in range(0, len(rec_seq), 3)]

        # Build record SEQRES string
        rec_str = ""
        line_res = [rec_residues[i : i + 13] for i in range(0, len(rec_residues), 13)]
        for i, res_group in enumerate(line_res):
            rec_str += "  SEQRES "
            rec_seq += " " * (3 - len(str(i))) + str(i)
            rec_str += " {}"
            rec_str += " " + " " * (3 - len(str(rec_len))) + str(rec_len)
            rec_str += "  " + " ".join(res_group)
            rec_str += "\n"

        rec_str = "".join([rec_str.format(*([c] * len(line_res))) for c in rec_chains])
        seqres_str += rec_str

    with args.out_file.open("w") as fp:
        fp.write(seqres_str)


if __name__ == "__main__":
    main()
