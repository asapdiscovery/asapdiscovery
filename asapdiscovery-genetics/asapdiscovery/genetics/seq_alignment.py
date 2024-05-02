import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO

# BioPython
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.glyphs import Rect, Text

# Bokeh imports
from bokeh.plotting import figure, output_file, save


class Alignment:
    def __init__(self, blast_match: pd.DataFrame, query: str, dir_save: Path):
        """An alignment object

        Parameters
        ----------
        blast_match : pd.DataFrame
            DataFrame with BLAST results
        query : str
            Descriptor of query sequence
        dir_save : Path
            Path for directory where results will be saved
        """
        self.dir_save = dir_save
        self.blast_query = query
        self.query_label = self.blast_query.strip().replace(" ", "_")

        df = blast_match.loc[blast_match["query"] == self.blast_query]
        self.query_matches = df
        self.seqs = self.query_matches["sequence"].to_numpy()
        self.ids = self.query_matches["ID"].to_numpy()
        self.descripts = self.query_matches["description"].to_numpy()
        self.hosts = self.query_matches["host"].to_numpy()
        self.organisms = self.query_matches["organism"].to_numpy()
        return

    @staticmethod
    def select_checkbox():
        raise NotImplementedError(
            "I'm not sure this can be implemented outside of Jupyter"
        )

    def select_keyword(self, match_string: str, selection_file: str):
        # First filter unique entries
        unique_idxs = np.unique(self.seqs, return_index=True)[1]
        ordered_idxs = np.sort(unique_idxs)
        unique_seqs = [self.seqs[i] for i in ordered_idxs]
        unique_ids = [self.ids[i] for i in ordered_idxs]
        unique_descp = [self.descripts[i] for i in ordered_idxs]

        # Filter sequences by keyword
        substrings = [
            match_string.capitalize(),
            match_string.lower(),
            match_string.upper(),
        ]

        filtered_idxs = [
            idx
            for idx, descp in enumerate(unique_descp)
            if any(substring in descp for substring in substrings)
        ]
        filtered_ids = [unique_ids[i] for i in filtered_idxs]
        filtered_seqs = [unique_seqs[i] for i in filtered_idxs]
        filtered_descp = [unique_descp[i] for i in filtered_idxs]

        if len(filtered_seqs) > 0:
            if unique_seqs[0] != filtered_seqs[0]:
                filtered_seqs = [unique_seqs[0]] + filtered_seqs
                filtered_descp = [unique_descp[0]] + filtered_descp
                filtered_ids = [unique_ids[0]] + filtered_ids
        else:
            print("The keyword provided didn't return any matches")
            filtered_seqs = [unique_seqs[0]]
            filtered_descp = [unique_descp[0]]
            filtered_ids = [unique_ids[0]]

        self.seqs = filtered_seqs
        self.ids = filtered_ids
        self.descripts = filtered_descp

        records = []
        for r in range(len(self.ids)):
            rec = SeqRecord(
                Seq(self.seqs[r]), id=self.ids[r], description=self.descripts[r]
            )
            records.append(rec)

        self.seq_records = selection_file
        SeqIO.write(records, self.seq_records, "fasta")

        return selection_file

    def select_taxonomy(self, match_string: str, selection_file: str):
        # First filter unique entries
        unique_idxs = np.unique(self.seqs, return_index=True)[1]
        ordered_idxs = np.sort(unique_idxs)

        # Extract info from provided match string
        or_querys = match_string.split("OR")
        host_str = "xxxxx"
        org_str = "xxxxx"
        for q in or_querys:
            if "host" in q:
                host_str = " ".join(q.strip().split(" ")[1:])
            elif "organism" in q:
                org_str = " ".join(q.strip().split(" ")[1:])

        filtered_idxs = [
            idx
            for idx in ordered_idxs
            if any(
                [
                    host_str.casefold() in self.hosts[idx].casefold(),
                    org_str.casefold() in self.organisms[idx].casefold(),
                ]
            )
        ]
        filtered_ids = [self.ids[i] for i in filtered_idxs]
        filtered_seqs = [self.seqs[i] for i in filtered_idxs]
        filtered_descp = [self.descripts[i] for i in filtered_idxs]

        if len(filtered_seqs) > 0:
            if self.seqs[0] != filtered_seqs[0]:
                filtered_seqs = [self.seqs[0]] + filtered_seqs
                filtered_descp = [self.descripts[0]] + filtered_descp
                filtered_ids = [self.ids[0]] + filtered_ids
        else:
            print("The keyword provided didn't return any matches")
            filtered_seqs = [self.seqs[0]]
            filtered_descp = [self.descripts[0]]
            filtered_ids = [self.ids[0]]

        self.seqs = filtered_seqs
        self.ids = filtered_ids
        self.descripts = filtered_descp

        records = []
        for r in range(len(self.ids)):
            rec = SeqRecord(
                Seq(self.seqs[r]), id=self.ids[r], description=self.descripts[r]
            )
            records.append(rec)

        self.seq_records = selection_file
        SeqIO.write(records, self.seq_records, "fasta")

        return selection_file

    def multi_seq_alignment(self, alignment_file):
        # Run alignment with MAFFT
        # SeqIO.write(self.seq_records, temp_file, "fasta")
        cmd = f"mafft {self.seq_records} > {alignment_file}"
        subprocess.run(cmd, shell=True, capture_output=True)

        self.align_obj = AlignIO.read(alignment_file, "fasta")
        return alignment_file

    def view_alignment(self, fontsize="9pt", plot_width=800, file_name="alignment"):
        """Bokeh sequence alignment view
        From: https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner"""
        # The function takes a biopython alignment object as input.
        aln = self.align_obj
        seqs = [rec.seq for rec in (aln)]  # Each sequence input
        text = [i for s in list(seqs) for i in s]  # Al units joind on same list

        # Shorten the description for display (string between the last [*])
        def matches(x):
            import re

            pattern = r"\[(.*?)\]"
            return re.findall(pattern, x)[-1]

        desc = [f"{matches(rec.description)} ({rec.id})" for rec in aln]

        # List with ALL colors
        colors = get_colors_protein(seqs)
        N = len(seqs[0])
        S = len(seqs)

        x = np.arange(1, N + 1)
        y = np.arange(0, S, 1)
        # creates a 2D grid of coords from the 1D arrays
        xx, yy = np.meshgrid(x, y)
        # flattens the arrays
        gx = xx.ravel()
        gy = yy.flatten()
        # use recty for rect coords with an offset
        recty = gy + 0.5
        # now we can create the ColumnDataSource with all the arrays
        print(f"Aligning {S} sequences of lenght {N}")
        # ColumnDataSource is a JSON dict that maps names to arrays of values
        source = ColumnDataSource(
            dict(x=gx, y=gy, recty=recty, text=text, colors=colors)
        )
        plot_height = len(seqs) * 10 + 50
        x_range = Range1d(0, N + 1, bounds="auto")  # (start, end)
        if N > 150:
            viewlen = 150
        else:
            viewlen = N
        # view_range is for the close up view
        view_range = (0, viewlen)
        tools = "xpan, xwheel_zoom, reset, save"

        # entire sequence view (no text, with zoom)
        p = figure(
            title=None,
            width=plot_width,
            height=plot_height,
            x_range=x_range,
            y_range=desc,
            tools=tools,
            min_border=0,
            toolbar_location="below",
        )
        # Rect simply places rectangles of wifth "width" into the positions defined by x and y
        rects = Rect(
            x="x",
            y="recty",
            width=1,
            height=1,
            fill_color="colors",
            line_color=None,
            fill_alpha=0.6,
        )
        # Source does mapping from keys in rects to values in ColumnDataSource definition
        p.add_glyph(source, rects)
        p.grid.visible = False
        p.xaxis.major_label_text_font_style = "bold"
        p.yaxis.major_label_text_font_size = "8pt"
        p.yaxis.minor_tick_line_width = 0
        p.yaxis.major_tick_line_width = 0

        # sequence text view with ability to scroll along x axis
        p1 = figure(
            title=None,
            width=plot_width,
            height=plot_height,
            x_range=view_range,
            y_range=desc,
            tools="xpan,reset",
            min_border=0,
            toolbar_location="below",
        )  # , lod_factor=1)
        # Text does the same thing as rectangles but placing letter (or words) instead, aligned accordingly
        glyph = Text(
            x="x",
            y="y",
            text="text",
            text_align="center",
            text_color="black",
            text_font_size=fontsize,
        )
        rects = Rect(
            x="x",
            y="recty",
            width=1,
            height=1,
            fill_color="colors",
            line_color=None,
            fill_alpha=0.4,
        )
        p1.add_glyph(source, glyph)
        p1.add_glyph(source, rects)

        p1.grid.visible = True
        p1.xaxis.major_label_text_font_style = "bold"
        p1.yaxis.major_label_text_font_style = "bold"
        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0

        p = gridplot([[p], [p1]], toolbar_location="below")

        output_file(filename=f"{file_name}.html", title="Alignment result")
        save(p)

        return p, f"{file_name}.html"

    @staticmethod
    def fasta_align_data(input_alignment, output_file):
        """Modify sequences in multi-seq alignment to remove gap characters '-'"""
        alignment = SeqIO.parse(input_alignment, "fasta")
        filtered_sequences = []
        for rec in alignment:
            # Remove gap characters '-' from the sequence
            filtered_sequence = rec.seq.ungap(
                "-"
            )  # ''.join(char for char in rec.seq if char != '-')
            # Update SeqRecord with the filtered sequence
            filtered_seq_record = SeqRecord(
                filtered_sequence, id=rec.id, description=rec.description
            )
            filtered_sequences.append(filtered_seq_record)

        SeqIO.write(filtered_sequences, output_file, "fasta")
        return output_file

    @staticmethod
    def csv_align_data(input_alignment, output_file, n_chains):
        alignment = SeqIO.parse(input_alignment, "fasta")
        df = pd.DataFrame(columns=["id", "sequence"])
        for rec in alignment:
            label_parts = rec.id.split("|")[1].split(".")
            red_label = f"{label_parts[0]}_{label_parts[1]}"
            seq_print = str(rec.seq)
            if n_chains > 1:
                # ColabFold reads multimer chains separated by ":"
                seq_print = ":".join(seq_print * n_chains)
            dfi = pd.DataFrame.from_dict({"id": [red_label], "sequence": [seq_print]})
            df = pd.concat([df, dfi], ignore_index=True)
        df.to_csv(output_file, index=False)
        return output_file


def do_MSA(
    alignment: Alignment,
    select_mode: str,
    file_prefix: str,
    plot_width: int,
    n_chains: int,
):
    save_file = alignment.dir_save / file_prefix
    # Select sequeneces of interest
    if select_mode == "checkbox":
        select_file = alignment.select_checkbox()
    elif "host" in select_mode or "organism" in select_mode:
        select_file = alignment.select_taxonomy(select_mode, f"{save_file}.fasta")
    else:
        select_file = alignment.select_keyword(select_mode, f"{save_file}.fasta")

    print(f"A fasta file {select_file} have been generated with the selected sequences")

    # Do multisequence alignment
    align_fasta = alignment.multi_seq_alignment(f"{save_file}_alignment.fasta")
    print(
        f"A fasta file {align_fasta} have been generated with the multi-seq alignment"
    )

    # Save CSV for ColabFold step
    clean_csv = alignment.csv_align_data(select_file, f"{save_file}.csv", n_chains)
    print(f"A csv file {clean_csv} have been generated with the selected sequences")

    p, align_html = alignment.view_alignment(
        plot_width=plot_width, file_name=f"{save_file}_alignment"
    )
    print(f"A html file {align_html} have been generated with the aligned sequences")

    return select_file, p


# Defining colors for each protein residue
def get_colors_protein(seqs):
    """Make colors for bases in sequence

    Args:
        seqs (list, str): List or string with protein sequence

    Returns:
        list: List with colors
    """
    text = [i for s in list(seqs) for i in s]
    aa_colors = {
        "A": "red",  # Alanine
        "R": "blue",  # Arginine
        "N": "green",  # Asparagine
        "D": "yellow",  # Aspartic acid
        "C": "orange",  # Cysteine
        "Q": "purple",  # Glutamine
        "E": "cyan",  # Glutamic acid
        "G": "magenta",  # Glycine
        "H": "pink",  # Histidine
        "I": "brown",  # Isoleucine
        "L": "gray",  # Leucine
        "K": "lime",  # Lysine
        "M": "teal",  # Methionine
        "F": "navy",  # Phenylalanine
        "P": "olive",  # Proline
        "S": "maroon",  # Serine
        "T": "silver",  # Threonine
        "W": "gold",  # Tryptophan
        "Y": "skyblue",  # Tyrosine
        "V": "violet",  # Valine
        "-": "white",
    }
    colors = [aa_colors[i] for i in text]
    return colors
