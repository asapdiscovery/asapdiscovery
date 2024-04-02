from pathlib import Path
import numpy as np
from asapdiscovery.data.openeye import load_openeye_pdb, save_openeye_pdb
from asapdiscovery.modeling.modeling import superpose_molecule


def rmsd_alignment(target_pdb: str, 
    ref_pdb: str,  
    final_pdb: str,
    target_chain="A",
    ref_chain="A",
    ):
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
    float, str
       RMSD after alignment, Path to saved PDB
    """
    protein = load_openeye_pdb(target_pdb)
    ref_protein = load_openeye_pdb(ref_pdb)

    aligned_protein, rmsd = superpose_molecule(ref_protein, protein, ref_chain=ref_chain, mobile_chain=target_chain)
    pdb_aligned = save_openeye_pdb(aligned_protein, final_pdb)

    return rmsd, pdb_aligned

def select_best_colabfold(results_dir:str, 
    seq_name:str, 
    pdb_ref:str, 
    chain="A", 
    final_pdb="aligned_protein.pdb",):
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

    Returns
    -------
    
    float, str
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
        raise FileNotFoundError(f"A folder with ColbFold results {results_dir} does not exist")

    default_CF = "*_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_*.pdb"

    for file_path in results_dir.glob(seq_name + default_CF):
        pdb_to_compare = file_path 
        seed = str(pdb_to_compare).split('_')[-1].split('.')[0]
        rmsd, pdb = rmsd_alignment(pdb_to_compare, pdb_ref, final_pdb, chain, chain)
        rmsds.append(rmsd)
        seeds.append(seed)
        file_seed.append(file_path)
        print(f"RMSD for seed {seed} is {rmsd} A")

    if len(rmsds) == 0:
        print(f"The ColabFold directory {results_dir / seq_name} was empty.")
        return 0, ""
    min_rmsd = np.argmin(rmsds)
    min_rmsd_file = file_seed[min_rmsd]
    print(f"Seed with less RMSD is {seeds[min_rmsd]} with RMSD {rmsds[min_rmsd]} A")

    min_rmsd, final_pdb = rmsd_alignment(min_rmsd_file, pdb_ref, final_pdb, chain, chain)

    return min_rmsd, str(final_pdb)

def save_alignment_pymol(pdbs:list, 
                         labels: list,
                         reference:str,
                         session_save:str):
    """Imports the provided PDBs into a Pymol session and saves

    Parameters
    ----------
    pdbs : list
        List with paths to pdb file to include
    labels : list
        List with labels of proteins
    reference : str
        Path to reference PDB
    session_save : str
        Path to save the PyMOL session
    """
    import pymol2

    p = pymol2.PyMOL()
    p.start()

    p.cmd.load(reference, object="reference_protein")

    for i, pdb in enumerate(pdbs):
        if len(pdb) > 0: 
            pname = "Protein_" + labels[i]
            p.cmd.load(pdb, object=pname)
            # PDBs should be aligned but in case they are not
            p.cmd.align(pname, "reference_protein")

    # set visualization
    p.cmd.set("bg_rgb", "white")
    p.cmd.bg_color("white")
    p.cmd.hide("everything")
    p.cmd.show("cartoon")
    p.cmd.set("transparency", 0.3)

    p.cmd.save(session_save)
    return 
