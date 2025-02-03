from pathlib import Path

import numpy as np
import pymol2
from asapdiscovery.data.backend.openeye import load_openeye_pdb, save_openeye_pdb
from asapdiscovery.modeling.modeling import superpose_molecule


def rmsd_alignment(
    target_pdb: str,
    ref_pdb: str,
    final_pdb: str,
    target_chain="A",
    ref_chain="A",
) -> tuple[float, Path]:
    """Calculate RMSD of a molecule against a reference

    Parameters
    ----------
    target_pdb : str
        Path to PDB of protein to align.
    ref_pdb : str
        Path to PDB to align target against.
    final_pdb : str
        Path to save PDB of aligned target.
    target_chain : str, optional
        The chain of target which will be used for alignment, by default "A"
    ref_chain : str, optional
        The chain of reference which will be used for alignment, by default "A"

    Returns
    -------
    float, Path
       RMSD after alignment, Path to saved PDB
    """
    protein = load_openeye_pdb(target_pdb)
    ref_protein = load_openeye_pdb(ref_pdb)

    aligned_protein, rmsd = superpose_molecule(
        ref_protein, protein, ref_chain=ref_chain, mobile_chain=target_chain
    )
    pdb_aligned = save_openeye_pdb(aligned_protein, final_pdb)

    return rmsd, pdb_aligned


def select_best_colabfold(
    results_dir: str,
    seq_name: str,
    pdb_ref: str,
    chain="A",
    final_pdb="aligned_protein.pdb",
    fold_model="alphafold2_ptm",
) -> tuple[float, Path]:
    """Select the best seed output (repetition) from a ColabFold run based on its RMSD wrt the reference.

    Parameters
    ----------
    results_dir : str
        The directory containing the ColabFold results.
    seq_name : str
        The name we gave to the sequence in the csv file.
    pdb_ref : str
        The path to the PDB of te reference protein.
    chain : str, optional
        Chain of both reference and generated PDB that will be used, by default "A"
    final_pdb : str, optional
        Path to the PDB where aligned structure will be saved, by default "aligned_protein.pdb"
    fold_model : str, optional
        The model used for ColabFold, by default "alphafold2_ptm"

    Returns
    -------
    Tuple[float, Path]
        RMSD after alignment, Path to saved PDB

    Raises
    ------
    FileNotFoundError
        The directory with ColabFold results does not exist
    """

    rmsds = []
    seeds = []
    file_seed = []

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"A folder with ColbFold results {results_dir} does not exist"
        )

    for file_path in results_dir.glob(f"{seq_name}*_{fold_model}_model_1_seed_*.pdb"):
        pdb_to_compare = file_path
        seed = str(pdb_to_compare).split("_")[-1].split(".")[0]
        rmsd, pdb = rmsd_alignment(pdb_to_compare, pdb_ref, final_pdb, chain, chain)
        rmsds.append(rmsd)
        seeds.append(seed)
        file_seed.append(file_path)
        print(f"RMSD for seed {seed} is {rmsd} A")

    if len(rmsds) == 0:
        print(f"No ColabFold entry for {seq_name} and model {fold_model} found.")
        return 0, ""
    min_rmsd = np.argmin(rmsds)
    min_rmsd_file = file_seed[min_rmsd]
    print(
        f"{seq_name} seed with least RMSD is {seeds[min_rmsd]} with RMSD {rmsds[min_rmsd]} A"
    )

    min_rmsd, final_pdb = rmsd_alignment(
        min_rmsd_file, pdb_ref, final_pdb, chain, chain
    )

    return min_rmsd, str(final_pdb)


def save_alignment_pymol(
    pdbs: list,
    labels: list,
    reference: str,
    session_save: str,
    align_chain=str,
    color_by_rmsd=False,
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
    session_save : str
        File name for the saved PyMOL session.
    align_chain : str
        Chain of ref to align target with.
    color_by_rmsd : bool, optional
        Option to color aligned targets by RMSD with respect to reference.
    """

    def hide_chain(p, chain, obj):
        """Hide the other chain from visualization in obj"""
        dimer_chains = {"A", "B"}
        hide_chain = (dimer_chains - {chain}).pop()
        p.cmd.select("chainb", f"{obj} and chain {hide_chain.upper()}")
        p.cmd.remove("chainb")
        p.cmd.delete("chainb")

    p = pymol2.PyMOL()
    p.start()

    p.cmd.load(reference, object="ref_protein")
    p.cmd.color("gray", "ref_protein")
    # Optionaly remove other chain from the reference protein
    if align_chain == "both":
        align_sel = ""
    else:
        align_sel = f" and chain {align_chain}"
        hide_chain(p, align_chain, "ref_protein")

    p.cmd.select("chaina", f"ref_protein{align_sel}")
    p.cmd.color("gray", "ref_protein")

    for i, pdb in enumerate(pdbs):
        if len(str(pdb)) > 0:
            # In case the entry is empty (when no CF output was found)
            pname = labels[i]
            p.cmd.load(pdb, object=pname)
            # PDBs should be aligned but in case they are not
            p.cmd.select("chainp", pname + align_sel)
            # It's better to align wrt a single chain than the whole protein (at least one binding site to compare)
            p.cmd.align(f"{pname} and chain A", "ref_protein and chain A")
            if color_by_rmsd:
                colorbyrmsd(p, "chainp", "chaina", minimum=0, maximum=2)
                p.cmd.color("red", "ref_protein")
            if len(align_chain) == 1:
                hide_chain(p, align_chain, pname)
            p.cmd.delete("chainp")
    p.cmd.delete("chaina")

    # set visualization
    p.cmd.set("bg_rgb", "white")
    p.cmd.bg_color("white")
    p.cmd.hide("everything")
    p.cmd.show("cartoon")
    p.cmd.set("transparency", 0.8)
    p.cmd.set("transparency", 0.3, "ref_protein")

    # Color ligand and binding site
    p.cmd.select("ligand", "resn UNK or resn LIG")
    p.cmd.select(
        "binding_site", "name CA within 5 of resn UNK or name CA within 5 resn LIG"
    )
    p.cmd.show("sticks", "ligand")
    p.cmd.color("red", "ligand")

    p.cmd.save(session_save)
    return


def colorbyrmsd(
    p: pymol2.PyMOL,
    target_sel: str,
    ref_sel: str,
    quiet=True,
    minimum=None,
    maximum=None,
):
    """Color aligned proteins by RMSD with respect to the target.
    Based on script by original authors Shivender Shandilya and Jason Vertrees,
    rewrite by Thomas Holder. License: BSD-2-Clause.
    http://pymolwiki.org/index.php/ColorByRMSD

    Parameters
    ----------
    p : pymol2.PyMOL
        Pymol session
    target_sel : str
        Selection of aligned target
    ref_sel : str
        Selection of reference protein
    quiet : bool, optional
        Not print RMSD info, by default True
    minimum : Union[int,float], optional
        Set a fixed min RMSD for coloring, by default None
    maximum : Union[int,float], optional
        Set a fixed max RMSD for coloring, by default None
    """
    from chempy import cpv

    selboth, aln = "both", "aln"
    p.cmd.align(target_sel, ref_sel, cycles=0, transform=0, object=aln)
    p.cmd.select(selboth, f"{target_sel} or {ref_sel}")

    idx2coords = {}
    p.cmd.iterate_state(
        -1, selboth, "idx2coords[model,index] = (x,y,z)", space=locals()
    )

    if p.cmd.count_atoms("?" + aln, 1, 1) == 0:
        p.cmd.refresh()

    b_dict = {}
    for col in p.cmd.get_raw_alignment(aln):
        assert len(col) == 2
        b = cpv.distance(idx2coords[col[0]], idx2coords[col[1]])
        for idx in col:
            b_dict[idx] = b

    p.cmd.alter(selboth, "b = b_dict.get((model, index), -1)", space=locals())

    p.cmd.orient(selboth)
    p.cmd.show_as("cartoon", "byobj " + selboth)
    p.cmd.color("gray", selboth)
    p.cmd.spectrum("b", "red_blue", selboth + " and b > -0.5", minimum, maximum)

    # Make colorbar
    if minimum is not None and maximum is not None:
        p.cmd.ramp_new("colorbar", "none", [minimum, maximum], ["red", "blue"])

    if not quiet:
        print("ColorByRMSD: Minimum Distance: %.2f" % (min(b_dict.values())))
        print("ColorByRMSD: Maximum Distance: %.2f" % (max(b_dict.values())))
        print(
            "ColorByRMSD: Average Distance: %.2f" % (sum(b_dict.values()) / len(b_dict))
        )

    p.cmd.delete(aln)
    p.cmd.delete(selboth)

    return
