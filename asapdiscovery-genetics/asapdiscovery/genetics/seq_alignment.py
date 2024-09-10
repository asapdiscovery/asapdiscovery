import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO

# BioPython
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
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

        self.sucess = False
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
        substrings = []
        for s in match_string.split(","):
            substrings.append(s)
            substrings.append(s.capitalize())
            substrings.append(s.lower())
            substrings.append(s.upper())
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
        filtered_hosts = [self.hosts[i] for i in filtered_idxs]
        filtered_orgs = [self.organisms[i] for i in filtered_idxs]

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
        self.hosts = filtered_hosts
        self.organisms = filtered_orgs

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

    def view_alignment(
        self,
        fontsize="11pt",
        plot_width=800,
        file_name="alignment",
        color_by_group=False,
        start_idx=0,
        skip=4,
        max_missmatch=2,
    ):
        """ "Bokeh sequence alignment view
            From: https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner

        Parameters
        ----------
        fontsize : str, optional
            Size of aminoacid one-letter IDs, by default "11pt"
        plot_width : int, optional
            width of alignment plot, by default 800
        file_name : str, optional
            suffix for html file, by default "alignment"
        color_by_group : bool, optional
            View mode where matching aminoacids are colored, by default False
        start_idx : int, optional
            Index of first aminiacid of reference sequence, by default 0
        skip : int, optional
            Skip for displayed indexes of reference sequence , by default 4
        max_missmatch : int, optional
            How many missmatches are tolerated for highlighted group match, by default 2

        Returns
        -------
        (bokeh.Column, str)
            Bokeh Column of layouts, path to saved html file.
        """

        # The function takes a biopython alignment object as input.
        aln = self.align_obj
        seqs = [rec.seq for rec in (aln)]  # Each sequence input
        text = [i for s in list(seqs) for i in s]  # Al units joind on same list

        N = len(seqs[0])
        S = len(seqs)

        # Shorten the description for display (string between the last [*])
        def matches(x):
            import re

            pattern = r"\[(.*?)\]"
            return re.findall(pattern, x)[-1]

        desc = [f"{matches(rec.description)} ({rec.id})" for rec in aln]

        # List with ALL colors
        # By aminoacid group or exact match
        if color_by_group:
            col_colors = []
            font_colors = []
            for col in range(N):  # Go through each column
                # Note: AlignIO item retrieval is done through a get_item function, so this has to be done with a loop
                col_string = aln[:, col]
                color, font_color = get_colors_by_aa_group(col_string, max_missmatch)
                col_colors.append(color)
                font_colors.append(font_color)
            colors = col_colors * S
            # Append each font_color list "colum-wise"
            font_colors = np.array(font_colors).T.flatten()
        else:
            colors = get_colors_protein(seqs)
            font_colors = ["black"] * len(colors)

        # Defining x indexes only for non-gap characters of ref sequence (seqs[0])
        seq_array = np.array(list(seqs[0]))
        x_non_gap = np.full(len(seqs[0]), " ", dtype="<U3")
        non_gap_idx = np.where(seq_array != "-")[0]
        current_idx = start_idx
        x_non_gap_locs = []
        # Iterate to indexes (this way we skip the gaps in the middle)
        for idx in non_gap_idx:
            if idx in non_gap_idx[::skip]:  # Skips every given index
                x_non_gap[idx] = str(current_idx)
                x_non_gap_locs.append(idx)
            current_idx += 1

        x = np.arange(0, N)
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
        x_range = Range1d(gx[0] - 1, N + 1, bounds="auto")  # (start, end)
        if N > 150:
            viewlen = 150
        else:
            viewlen = N
        # view_range is for the close up view
        view_range = (gx[0] - 1, viewlen)
        tools = "xpan, xwheel_zoom, reset, save"

        # entire sequence view (no text, with zoom)
        p1 = figure(
            title=None,
            width=plot_width,
            height=plot_height,
            x_range=x_range,
            y_range=desc,
            tools=tools,
            min_border=0,
        )
        p1.toolbar_location = None
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
        p1.add_glyph(source, rects)
        p1.grid.visible = False
        p1.xaxis.major_label_text_font_style = "bold"
        p1.yaxis.major_label_text_font_size = "8pt"
        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0

        plot_height = len(seqs) * 20 + 30
        # sequence text view with ability to scroll along x axis
        p2 = figure(
            title=None,
            width=plot_width,
            height=plot_height,
            x_range=view_range,
            y_range=desc,
            tools=tools,
            min_border=0,
            toolbar_location="below",
        )  # , lod_factor=1)
        # Text does the same thing as rectangles but placing letter (or words) instead, aligned accordingly
        text_source = ColumnDataSource(
            dict(x=gx, y=gy, recty=recty, text=text, colors=font_colors)
        )
        glyph = Text(
            x="x",
            y="y",
            text="text",
            text_color="colors",
            text_align="center",
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
        # Blank plot to hold the position labels
        p_blank = figure(
            width=plot_width,
            height=40,
            x_range=view_range,
            y_range=Range1d(0, 1),
            title=None,
            toolbar_location=None,
            tools="",
            outline_line_alpha=0,
        )
        p_blank.xaxis.visible = False
        p_blank.yaxis.visible = False
        p_blank.grid.visible = False
        label_source = ColumnDataSource(dict(x=x, y=[0.05] * len(x), text=x_non_gap))
        labels = Text(
            x="x",
            y="y",
            text="text",
            text_color="black",
            text_align="center",
            text_font_size=str(int(fontsize[:-2]) - 2) + "pt",
        )
        p2.add_glyph(text_source, glyph)
        p2.add_glyph(source, rects)
        p_blank.add_glyph(label_source, labels)

        view_range = Range1d(gx[0] - 1, viewlen)
        p2.grid.visible = True
        p2.xaxis.major_label_text_font_style = "bold"
        p2.yaxis.major_label_text_font_style = "bold"
        p2.yaxis.minor_tick_line_width = 0
        p2.yaxis.major_tick_line_width = 0
        p2.xaxis.major_label_text_font_size = "0pt"
        p2.add_layout(
            LinearAxis(major_label_text_font_size="0pt", ticker=list(x_non_gap_locs)),
            "above",
        )
        p2.x_range = view_range
        p_blank.x_range = view_range

        p = column(p1, p_blank, p2)

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
                seq_print = ":".join([seq_print] * n_chains)
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
    color_by_group: bool,
    start_alignment_idx: int,
):
    save_file = alignment.dir_save / file_prefix
    # Select sequeneces of interest
    if select_mode == "checkbox":
        select_file = alignment.select_checkbox()
    elif "host" in select_mode or "organism" in select_mode:
        select_file = alignment.select_taxonomy(select_mode, f"{save_file}.fasta")
    else:
        select_file = alignment.select_keyword(select_mode, f"{save_file}.fasta")

    alignment.select_file = select_file
    print(
        f"A fasta file {alignment.select_file} have been generated with the selected sequences"
    )

    # Do multisequence alignment
    align_fasta = alignment.multi_seq_alignment(f"{save_file}_alignment.fasta")
    alignment.align_file = align_fasta
    print(
        f"A fasta file {alignment.align_file} have been generated with the multi-seq alignment"
    )

    # Save CSV for ColabFold step
    clean_csv = alignment.csv_align_data(
        alignment.select_file, f"{save_file}.csv", n_chains
    )
    print(f"A csv file {clean_csv} have been generated with the selected sequences")

    p, align_html = alignment.view_alignment(
        plot_width=plot_width,
        file_name=f"{save_file}_alignment",
        color_by_group=color_by_group,
        start_idx=start_alignment_idx,
    )
    print(f"A html file {align_html} have been generated with the aligned sequences")

    alignment.sucess = True
    return alignment


# Defining colors for each protein residue
def get_colors_protein(seqs):
    """Make colors for bases in sequence

    Parameters
    ----------
    seq : List[str]
        List with protein sequences

    Returns
    -------
    list[str]
        List with colors
    """
    # List of all aminoacids in the list of seqs, in the same list
    text = [i for s in list(seqs) for i in s]
    colors = []
    for aa in text:
        amino = AAcid(aa)
        colors.append(amino.get_aminoacid_color())
    return colors


# Defining colors for each protein residue
def get_colors_by_aa_group(seq: str, max_missmatch=2):
    """Make fill and text color for exact and group aminoacid matches

    Parameters
    ----------
    seq : str
        String with protein sequence
    max_missmatch : int, optional
       Maximum number of group missmatches after which match won't be highlighted, by default 2

    Returns
    -------
    str, list[str]
        Fill color, Font colors
    """

    from collections import Counter

    seq_len = len(seq)
    aa_groups = [AAcid(aa).get_aminoacid_group() for aa in seq]
    aa_counts = Counter(aa_groups)
    max_group, max_group_count = aa_counts.most_common(1)[0]
    # Default font color
    font_color = ["black"] * seq_len
    # Check the case where all aa's are the same
    if seq == seq_len * seq[0]:
        color = "red"
    # Check the case where all aa's belong to the same group (with some max missmatches)
    elif max_group_count >= seq_len - max_missmatch:
        if max_group is None:  # In case most "matches" are gaps
            color = "white"
        else:
            color = "yellow"
            # Make font red for missmatches
            font_color = ["black" if item == max_group else "red" for item in aa_groups]
    else:
        color = "white"
    return color, font_color


class AAcid:
    def __init__(self, letter_id):
        """An aminoacid object

        Parameters
        ----------
        letter_id : str
            Amino Acid one or three-letter identifier
        """
        from MDAnalysis.lib.util import convert_aa_code

        # An empty aminoacid
        if letter_id == "-":
            self.one_letter_id = letter_id
            self.three_letter_id = None
        elif len(letter_id) == 1:
            self.one_letter_id = letter_id
            self.three_letter_id = convert_aa_code(letter_id)
        elif len(letter_id == 3):
            self.three_letter_id = letter_id
            self.one_letter_id = convert_aa_code(letter_id)
        else:
            raise ValueError(
                "The input must be either the aminoacid 1 or 3-letter code"
            )

    def get_aminoacid_group(self):
        agroups = {
            "aliphatic": ["A", "V", "I", "L", "M"],
            "aromatic": ["F", "W", "Y"],
            "neutral": ["N", "Q", "S", "T"],
            "acidic": ["D", "E"],
            "basic": ["R", "H", "K"],
            "cys": ["C"],
            "gly": ["G"],
            "pro": ["P"],
        }
        for key in agroups:
            if self.one_letter_id in agroups[key]:
                return key
        return

    def get_aminoacid_color(self):
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
        return aa_colors[self.one_letter_id]
