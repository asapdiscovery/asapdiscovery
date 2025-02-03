import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pymol2
from asapdiscovery.spectrum.blast import pdb_to_seq
from asapdiscovery.spectrum.seq_alignment import get_colors_by_aa_group
from Bio import Align, AlignIO, pairwise2


def pairwise_alignment(pdb_file, pdb_align, start_idxA, start_idxB):
    """Align pdb_file and pdb_align based on pairwise seq alignment"""
    pdb1 = Path(pdb_file)
    pdb2 = Path(pdb_align)

    # Chain A
    rec1 = pdb_to_seq(pdb1, chain="A")
    rec2 = pdb_to_seq(pdb2, chain="A")
    seq1 = str(rec1.seq).replace("X", "")
    seq2 = str(rec2.seq).replace("X", "")
    alignmentsA = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.8, -0.5)[0]
    print("Pwise alignment for Chain A", "\n", pairwise2.format_alignment(*alignmentsA))

    # Chain B
    rec1 = pdb_to_seq(pdb1, chain="B")
    rec2 = pdb_to_seq(pdb2, chain="B")
    seq1 = str(rec1.seq).replace("X", "")
    seq2 = str(rec2.seq).replace("X", "")
    alignmentsB = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.8, -0.5)[0]
    print("Pwise alignment for Chain B", "\n", pairwise2.format_alignment(*alignmentsB))

    colorsA = get_colors_pairwise(alignmentsA, start_idxA)
    colorsB = get_colors_pairwise(alignmentsB, start_idxB)
    pdb_align = [pdb_align]

    return pdb_align, colorsA, colorsB


def fasta_alignment(
    fasta_a,
    fasta_b,
    fasta_sel,
    pdb_labels,
    start_idxA,
    start_idxB,
    pdb_align=None,
    struct_dir=None,
    max_mismatches=0,
):
    """Align ref pdb_file with pdb_align or pdbs in struct_dir based on multi-seq alignment"""
    alignmentsA = AlignIO.read(fasta_a, format="fasta")
    alignmentsB = AlignIO.read(fasta_b, format="fasta")
    if len(alignmentsA) != len(alignmentsB):
        raise ValueError(
            "The fasta alignments for Chains A and B must have the same entries."
        )
    if len(alignmentsA) < 2 or len(alignmentsB) < 2:
        raise ValueError(
            "Each fasta file must AT LEAST contain a reference and one sequence to align."
        )
    fasta_sel = list(map(int, fasta_sel.split(",")))

    def check_for_trailing_aas(pdb, aln_entry, chain):
        """Check if there's trailing AA's at the beggining of alignment"""
        rec = pdb_to_seq(pdb, chain=chain)
        first_aa = rec.seq[0]
        start_add = aln_entry.seq.find(first_aa)
        return start_add

    if pdb_align is not None:
        # 2-protein alignment mode
        print(f"Performing a 2-protein alignment of {pdb_align} to reference.")
        if len(fasta_sel) != 2:
            raise ValueError(
                "the fasta_sel variable must be a comma-separated list of 2 indexes for the 1-pdb alignment mode."
            )
        if len(alignmentsA) > 2 and fasta_sel == [0, 1]:
            warnings.warn(
                "The default of fasta_sel '0,1' is being used with an alignment>2. You may want to set the sel manually."
            )
        extra_start_idxA = check_for_trailing_aas(
            Path(pdb_align), alignmentsA[fasta_sel[1]], "A"
        )
        extra_start_idxB = check_for_trailing_aas(
            Path(pdb_align), alignmentsB[fasta_sel[1]], "B"
        )
        start_idxA -= extra_start_idxA
        start_idxB -= extra_start_idxB
        pdb_align = [pdb_align]

    elif struct_dir is not None:
        # Multi-protein alignment mode
        print(
            f"Performing a multi-protein alignment from folder {struct_dir} to reference."
        )
        pdb_align = []
        labels = ["ref_protein"]
        aln_sel = [0]  # Make ref to be always on the top
        for seq_entry in alignmentsA:
            # loop over alignment file:
            seq_name = seq_entry.name.split("|")[1].split(".")[0]
            f_name = list(Path(struct_dir).glob(f"{seq_name}*.pdb"))
            if len(f_name) == 0:
                print(f"Seq entry for {seq_name} didn't have a PDB in {struct_dir}")
                continue
            print(f"Reading structure {f_name[0].stem}, for seq {seq_name}")
            labels.append(f_name[0].stem)
            pdb_align.append(str(f_name[0]))
        pdb_alignB = []
        for seq_entry in alignmentsB:
            seq_name = seq_entry.name.split("|")[1].split(".")[0]
            f_name = list(Path(struct_dir).glob(f"{seq_name}*.pdb"))
            if len(f_name) == 0:
                continue
            pdb_alignB.append(str(f_name[0]))
        if pdb_align != pdb_alignB:
            raise ValueError(
                "The fasta files for Chains A and B did not have the same entries! Make sure they do."
            )
        aln_sel = list(range(len(pdb_align)))
        if len(pdb_align) == 0:
            # In this case, the entry names may not match the pdb names, so we try to match by sequence
            pdb_align, aln_sel, labels = get_idx_by_seq(struct_dir, alignmentsA)
        # Checking validity of provide parameters
        if len(pdb_labels) < len(labels):
            pdb_labels = labels
            warnings.warn(
                "No pdb_label parameter was provided or it was given incorrectly (less PyMOL labels than PDBs). Will determine automatically."
            )
        if len(fasta_sel) == len(pdb_align):
            warnings.warn(
                "You provided a list of indexes for the fasta file. These can also be calculated if not provided, so make sure they are correct!"
            )
        else:
            if len(fasta_sel) > len(pdb_align):
                warnings.warn(
                    "More fasta indexes given than pdbs in directory! Will determine automatically."
                )
            fasta_sel = aln_sel
        # For now, the functionality of start_idx only works if all proteins have the same value
        extra_start_idxA = check_for_trailing_aas(
            Path(pdb_align[0]), alignmentsA[fasta_sel[1]], "A"
        )
        extra_start_idxB = check_for_trailing_aas(
            Path(pdb_align[0]), alignmentsB[fasta_sel[1]], "B"
        )
        start_idxA -= extra_start_idxA
        start_idxB -= extra_start_idxB

    colorsA = get_colors_multi(
        alignmentsA, fasta_sel, start_idxA, max_mismatch=max_mismatches
    )
    colorsB = get_colors_multi(
        alignmentsB, fasta_sel, start_idxB, max_mismatch=max_mismatches
    )
    return pdb_align, colorsA, colorsB, pdb_labels


def get_idx_by_seq(dir_path: str, alignments: Align.MultipleSeqAlignment):
    """Auxiliary function to match pdbs of a directory to entries on MultiSeq alignment object

    Parameters
    ----------
    dir_path : str
        Path to directory with PDBs
    alignments : Align.MultipleSeqAlignment
        BioPython multi sequence alignment with protein sequence entries

    Returns
    -------
    (list, list, list)
        Returns equal-length lists of pdbs, idx in fasta object and labels
    """
    pdb_seqs = []
    pdbs = []
    for pdb in Path(dir_path).glob("*.pdb"):
        rec = pdb_to_seq(pdb, chain="A")
        pdb_seq = str(rec.seq).replace("X", "")
        pdb_seqs.append(pdb_seq)
        pdbs.append(str(pdb))
    seq_idxs = []
    aln_sel = []
    labels = []
    for i, aln in enumerate(alignments):
        seq = str(aln.seq).replace("-", "")  # gap-less sequence in alignment
        seq_idx = [idx for idx, pdb_seq in enumerate(pdb_seqs) if seq in pdb_seq]
        if len(seq_idx) > 0:
            seq_idxs.append(seq_idx[0])
            aln_sel.append(i)
            labels.append(aln.name)
    pdb_align = np.array(pdbs)[seq_idxs]
    return pdb_align, aln_sel, labels


def get_colors_pairwise(alignment, start_idx=0):
    seqs = [alignment.seqA, alignment.seqB]
    N = len(seqs[-1])
    columns = list(zip(*seqs))
    col_colors = []
    i = 0
    for col in range(N):  # Go through each column
        col_string = "".join(columns[col])
        colors_dict = {"exact": "white", "group": "orange", "none": "red"}
        color, font_color, key = get_colors_by_aa_group(col_string, 0, colors_dict)
        if col_string[1] != "-":
            col_colors.append(color)
            i += 1
    color_dict = {
        (index + start_idx): string for index, string in enumerate(col_colors)
    }
    return color_dict


def get_colors_multi(alignment, seq_idx, start_idx, max_mismatch):
    seqs = [rec.seq for rec in (alignment)]  # Each sequence input

    N = len(seqs[-1])
    col_colors = []
    i = 0
    for col in range(N):  # Go through each column
        col_string = alignment[:, col]
        col_string_cut = "".join(col_string[i] for i in seq_idx)
        colors_dict = {"exact": "white", "group": "orange", "none": "red"}
        color, font_color, key = get_colors_by_aa_group(
            col_string_cut, max_mismatch, colors_dict
        )
        if col_string[1] != "-":
            col_colors.append(color)
            i += 1
    color_dict = {
        (index + start_idx): string for index, string in enumerate(col_colors)
    }
    return color_dict


def save_pymol_seq_align(
    pdbs: list,
    labels: list,
    reference: str,
    color_dict: Union[list[dict], dict],
    session_save: str,
) -> None:
    """Imports the provided PDBs into a Pymol session and saves

    Parameters
    ----------
    pdbs : list
       List with paths to pdb file to include.
    labels : list
        List with labels that will be used in protein objects.
    reference : str
        Path to reference PDB.
    color_dict : Union[List[dict], dict]
        Dictionary with colors for chain or list of dictionary for chains.
    session_save : str
        File name for the saved PyMOL session.
    """

    p = pymol2.PyMOL()
    p.start()
    # load ref protein
    p.cmd.load(reference, object=labels[0])
    p.cmd.color("gray", labels[0])

    # Load other pdbs
    for i, pdb in enumerate(pdbs):
        p.cmd.load(pdb, object=labels[i + 1])
        p.cmd.align(f"{labels[i + 1]} and chain A", f"{labels[0]} and chain A")
        p.cmd.color("gray", labels[i + 1])
        p.cmd.select("chaina", f"{labels[i+1]} and chain A")
        p.cmd.select("chainb", f"{labels[i+1]} and chain B")
        # chain A
        for idx, color in color_dict[0].items():
            p.cmd.color(f"{color}", f"chaina and resi {idx}")
        # chain B
        for idx, color in color_dict[1].items():
            p.cmd.color(f"{color}", f"chainb and resi {idx}")

    # set visualization
    p.cmd.set("bg_rgb", "white")
    p.cmd.bg_color("white")
    p.cmd.hide("everything")
    p.cmd.show("cartoon")
    p.cmd.set("transparency", 0.8)

    # Color ligand and binding site
    p.cmd.select("lig", "resn LIG or resn UNK")
    p.cmd.extract("ligand", "lig")
    p.cmd.show("sticks", "ligand")

    if len(pdbs) < 3:
        # surface display only for 2-pdb comparison
        p.cmd.show("surface", labels[1])
        p.cmd.set("transparency", 0.3)

    p.cmd.delete("chaina")
    p.cmd.delete("chainb")
    p.cmd.delete("lig")
    p.cmd.save(session_save)
    return
