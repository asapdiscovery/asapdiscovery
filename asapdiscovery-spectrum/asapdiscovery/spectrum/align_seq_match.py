from pathlib import Path
from typing import Union, List
import pymol2

from asapdiscovery.spectrum.seq_alignment import get_colors_by_aa_group
from asapdiscovery.spectrum.blast import pdb_to_seq
from Bio import pairwise2, AlignIO

import warnings

def pairwise_alignment(pdb_file, pdb_align, start_idxA, start_idxB):
    """ Align pdb_file and pdb_align based on pairwise seq alignment
    """
    pdb1 = Path(pdb_file)
    pdb2 = Path(pdb_align)

    #Chain A
    rec1 = pdb_to_seq(pdb1, chain="A")
    rec2 = pdb_to_seq(pdb2, chain="A")
    seq1 = str(rec1.seq).replace("X", "")
    seq2 = str(rec2.seq).replace("X", "")
    alignmentsA = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.8, -0.5)[0]
    print(pairwise2.format_alignment(*alignmentsA))

    # Chain B
    rec1 = pdb_to_seq(pdb1, chain="B")
    rec2 = pdb_to_seq(pdb2, chain="B")
    seq1 = str(rec1.seq).replace("X", "")
    seq2 = str(rec2.seq).replace("X", "")
    alignmentsB = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.8, -0.5)[0]
    print(pairwise2.format_alignment(*alignmentsB))

    colorsA = get_colors_pairwise(alignmentsA, start_idxA)
    colorsB = get_colors_pairwise(alignmentsB, start_idxB)
    pdb_align = [pdb_align]

    return pdb_align, colorsA, colorsB

def fasta_alignment(fasta_a,
                    fasta_b,
                    fasta_sel,
                    pdb_labels,
                    start_idxA,
                    start_idxB,
                    pdb_align=None,
                    struct_dir=None,
                    max_mismatches=0,
    ):
    """Align ref pdb_file with pdb_align or pdbs in struct_dir based on multi-seq alignment
    """
    alignmentsA = AlignIO.read(fasta_a, format='fasta')
    alignmentsB = AlignIO.read(fasta_b, format='fasta')
    assert len(alignmentsA) >= 2
    assert len(alignmentsB) >= 2

    fasta_sel = list(map(int, fasta_sel.split(",")))
    if pdb_align is not None:
        assert len(fasta_sel) == 2
        pdb_align = [pdb_align]
        if len(alignmentsA) > 2 and fasta_sel==[0,1]:
            warnings.warn("The default of fasta_sel '0,1' is being used with an alignment>2. You may want to set the sel manually.")
    elif struct_dir is not None:
        pdb_align = []
        labels = ["ref_protein"]
        fasta_sel = [0] # Make ref to be always on the top
        # loop over alignment file
        for seq_entry in alignmentsA:
            seq_name = seq_entry.name.split("|")[1].split(".")[0]
            f_name = list(Path(struct_dir).glob(f"{seq_name}*.pdb"))[0]
            if not f_name.exists():
                print(f"Seq entry for {seq_name} didn't have a PDB in {struct_dir}")
            print(f"Reading structure {f_name.stem}, for seq {seq_name}")
            labels.append(f_name.stem)
            pdb_align.append(str(f_name))
        fasta_sel = list(range(len(alignmentsA)))

        if len(pdb_labels) < len(labels):
            pdb_labels = labels
            print(f"Labels for PyMOL objs weren't properly provided. Will determine automatically.")
        if len(fasta_sel) < len(pdb_align):
            fasta_sel = list(range(len(pdb_align)))
            print(f"Fasta indexes weren't provided for all PDBs in directory. Will set as range.")
        if len(fasta_sel) > len(pdb_align):
            raise ValueError("More fasta indexes given than pdbs in directory!")
    colorsA = get_colors_multi(alignmentsA, fasta_sel, start_idxA, max_mismatch=max_mismatches)
    colorsB = get_colors_multi(alignmentsB, fasta_sel, start_idxB, max_mismatch=max_mismatches)

    return pdb_align, colorsA, colorsB, pdb_labels


def get_colors_pairwise(alignment, start_idx=0):

    seqs = [alignment.seqA, alignment.seqB]  # Each sequence input

    N = len(seqs[-1])
    columns = list(zip(*seqs))
    col_colors = []
    i = 0
    for col in range(N):  # Go through each column
        # Note: AlignIO item retrieval is done through a get_item function, so this has to be done with a loop
        col_string = ''.join(columns[col])
        colors_dict = {"exact": "white", "group": "orange", "none": "red"}
        color, font_color, key = get_colors_by_aa_group(col_string, 0, colors_dict)
        if col_string[0] != '-':
            col_colors.append(color)
            i += 1
    color_dict =  {(index+start_idx): string for index, string in enumerate(col_colors)}
    return color_dict

def get_colors_multi(alignment, seq_idx, start_idx, max_mismatch):

    seqs = [rec.seq for rec in (alignment)]  # Each sequence input

    N = len(seqs[-1])
    col_colors = []
    i = 0
    for col in range(N):  # Go through each column
        col_string = alignment[:, col]
        col_string_cut = ''.join(col_string[i] for i in seq_idx)
        colors_dict = {"exact": "white", "group": "orange", "none": "red"}
        color, font_color, key = get_colors_by_aa_group(col_string_cut, max_mismatch, colors_dict)
        if col_string[0] != '-':
            col_colors.append(color)
            i += 1
    color_dict =  {(index+start_idx): string for index, string in enumerate(col_colors)}
    return color_dict

def save_pymol_seq_align(
    pdbs: list, labels: list, reference: str, color_dict: Union[List[dict], dict], session_save: str,
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
        p.cmd.load(pdb, object=labels[i+1])
        p.cmd.align(labels[i+1], labels[0])
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
    p.cmd.select("ligand", "resn LIG")
    p.cmd.extract("ligand", "ligand")
    p.cmd.show("sticks", "ligand")

    if len(pdbs) < 3:
        # surface display only for 2-pdb comparison
        p.cmd.show("surface", labels[1])
        p.cmd.set("transparency", 0.3)

    p.cmd.delete("chaina")
    p.cmd.delete("chainb")
    p.cmd.delete("ligand")
    p.cmd.save(session_save)
    return
