import os.path

import yaml
from kinoml.databases.pdb import download_pdb_structure
# from kinoml.modeling.OEModeling import superpose_proteins
from openeye import oechem
from typing import Iterable
from MDAnalysis.coordinates.PDB import PDBReader, PDBWriter
# from MDAnalysis.analysis.align import alignto
# import mdtraj as md
# from mdtraj.formats import PDBTrajectoryFile





def load_pdbs_from_yaml(pdb_list_yaml):
    print(f"Loading pdb list from {pdb_list_yaml}...")
    with open(pdb_list_yaml, 'r') as f:
        pdb_list = yaml.safe_load(f)
    return pdb_list


def download_PDBs(pdb_list, pdb_path):
    """
    Downloads pdbs from pdb_list_yaml using Kinoml.

    Parameters
    ----------
    pdb_list_yaml
    pdb_path

    Returns
    -------

    """
    print(f"Downloading PDBs to {pdb_path}")
    for pdb in pdb_list:
        print(pdb)
        download_pdb_structure(pdb, pdb_path)

def pymol_alignment(pdb_path, ref_path, out_path):
    import pymol
    pymol.cmd.load(pdb_path, "mobile")
    pymol.cmd.load(ref_path, "ref")
    pymol.cmd.align("polymer and name CA and mobile and chain A",
                        "polymer and name CA and ref and chain A",
                    quiet=0)
    pymol.cmd.save(out_path, "mobile")
    pymol.cmd.delete("all")

def align_all_pdbs(pdb_list, pdb_dir_path, ref_path=None, ref_name=None):
    if not ref_path:
        ref = pdb_list[0]
        ref_path = os.path.join(pdb_dir_path, f'rcsb_{ref}.pdb')
    else:
        ref = ref_name
    for pdb in pdb_list:
        pdb_path = os.path.join(pdb_dir_path, f'rcsb_{pdb}.pdb')
        new_pdb_path = os.path.join(pdb_dir_path, f"{pdb}_aligned_to_{ref}.pdb")
        print(f"Aligning {pdb_path} \n"
              f"to {ref_path} \n"
              f"and saving to {new_pdb_path}")
        pymol_alignment(pdb_path, ref_path, new_pdb_path)

if __name__ == '__main__':
    pdb_list = load_pdbs_from_yaml('mers-structures.yaml')
    pdb_dir_path = '/Users/alexpayne/Scientific_Projects/mers-structures'
    download_PDBs(pdb_list, pdb_dir_path)
    # ref_path = '/Users/alexpayne/lilac-mount-point/fragalysis/extra_files/reference.pdb'
    ref_path = pdb_dir_path + "/rcsb_4RSP.pdb"

    align_all_pdbs(pdb_list,
                   pdb_dir_path,
                   # ref_path,
                   # ref_name="frag_ref_pymol"
                   )
