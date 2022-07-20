import os.path

import yaml
from kinoml.databases.pdb import download_pdb_structure
from kinoml.modeling.OEModeling import superpose_proteins
from openeye import oechem
# from MDAnalysis.coordinates.PDB import PDBReader, PDBWriter
# from MDAnalysis.analysis.align import alignto
import mdtraj as md
from mdtraj.formats import PDBTrajectoryFile

def load_pdbs_from_yaml(pdb_list_yaml):
    with open(pdb_list_yaml, 'r') as f:
        pdb_list = yaml.safe_load(f)
    return pdb_list


def download_PDBs(pdb_list_yaml, pdb_path):
    """
    Downloads pdbs from pdb_list_yaml using Kinoml.

    Parameters
    ----------
    pdb_list_yaml
    pdb_path

    Returns
    -------

    """
    ## First load the list of PDB structures
    pdb_list = load_pdbs_from_yaml(pdb_list_yaml)

    for pdb in pdb_list:
        download_pdb_structure(pdb, pdb_path)


# def align_pdb_to_reference(pdb_path, ref_path):
#     ## first load in pdb files using mdanalysis
#
#     pdb = md.load_pdb(pdb_path)
#     ref = md.load_pdb(ref_path)
#     pdb.superpose(ref, atom_indices=pdb.topology.select("chainid 0 and protein and backbone"))
#     pdb_list = os.path.split(pdb_path)
#     ref_name = os.path.basename(ref_path)
#     new_pdb_path = os.path.join(pdb_list[0], f"aligned_to_{ref_name}_{pdb_list[1]}")
#     # new_pdb = md.formats.PDBTrajectoryFile(new_pdb_path, mode='w')
#
#     pdb.save(new_pdb_path)

    # new_pdb.write(
    #     pdb.xyz,
    #     pdb.topology,
    # )
    # pdb = PDBTrajectoryFile(pdb_path)
    # ref = PDBTrajectoryFile(ref_path)
    #
    # pdb.superpose(ref, atom_indices=pdb.topology.select("chainid 0 and backbone"))


# (pdb, ref, select="protein and name CA and chain A", weights="mass")]


def load_openeye_mol(pdb_path):
    ifs = oechem.oemolistream()
    ifs.SetFormat(oechem.OEFormat_PDB)
    ifs.open(pdb_path)
    mols = []
    i = 0
    for mol in ifs.GetMolBases():
        print(i)
        i+=1
        mols.append(oechem.OEMol(mol))
    return mols[0]

# def align_pdb_to_reference(pdb_path, ref_path):

def align_all_pdbs(pdb_list, pdb_dir_path, ref_path=None, ref_name=None):
    if not ref_path:
        ref = pdb_list[0]
        ref_path = os.path.join(pdb_dir_path, f'rcsb_{ref}.pdb')
    else:
        ref = ref_name
    ref_mol = load_openeye_mol(ref_path)

    ofs = oechem.oemolostream()
    for pdb in pdb_list:
        pdb_path = os.path.join(pdb_dir_path, f'rcsb_{pdb}.pdb')
        pdb_mol = load_openeye_mol(pdb_path)
        pdb_mol.
        new_pdb_path = os.path.join(pdb_dir_path, f"{pdb}_aligned_to_{ref}.pdb")
        ofs.open(new_pdb_path)
        aligned_mol = superpose_proteins(ref_mol, pdb_mol, chain_id="A")
        oechem.OEWriteMolecule(ofs, aligned_mol)


if __name__ == '__main__':
    # download_PDBs('mers-structures.yaml', '/Users/alexpayne/lilac-mount-point/mers-structures')
    pdb_list = load_pdbs_from_yaml('mers-structures.yaml')
    mers_path = '/Users/alexpayne/lilac-mount-point/mers-structures'
    align_all_pdbs(pdb_list, mers_path, ref_path='/Users/alexpayne/lilac-mount-point/fragalysis/extra_files/reference.pdb', ref_name='frag_ref')
