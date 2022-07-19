from Bio import PDB
import yaml
import os
from kinoml.databases.pdb import download_pdb_structure


def download_PDBs(pdb_list_yaml, pdb_path):
    """
    Downloads the given PDBs from the yaml file to the directory specified in pdb_path

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

    print(pdb_list)

    ## Make sure PDB directory path exists
    if not os.path.exists(pdb_path):
        os.mkdir(pdb_path)

    ## Move to requested directory
    path_list = os.path.split(pdb_path)

    os.chdir(path_list[0])
    print(f"Downloading PDBs to {path_list[0]} in {path_list[1]}")

    ## Then use BioPython to download
    pdb_obj = PDB.PDBList()
    pdb_obj.download_pdb_files(pdb_list,
                               # file_format='pdb',
                               overwrite=False,
                               pdir=path_list[1])





if __name__ == '__main__':
    download_PDBs('mers-structures.yaml', '/Users/alexpayne/Scientific_Projects/mers-drug-discovery/mers-structures')
    # download_pdb_structure('4RSP', '/Users/alexpayne/Scientific_Projects/mers-drug-discovery/mers-structures')