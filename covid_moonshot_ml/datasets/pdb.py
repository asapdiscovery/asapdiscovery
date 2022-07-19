import yaml
from kinoml.databases.pdb import download_pdb_structure

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
    with open(pdb_list_yaml, 'r') as f:
        pdb_list = yaml.safe_load(f)

    for pdb in pdb_list:
        download_pdb_structure(pdb, pdb_path)


if __name__ == '__main__':
    download_PDBs('mers-structures.yaml', '/Users/alexpayne/Scientific_Projects/mers-drug-discovery/mers-structures')