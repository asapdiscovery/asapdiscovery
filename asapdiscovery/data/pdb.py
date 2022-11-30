def load_pdbs_from_yaml(pdb_list_yaml):
    """
    Load a pdb list from yaml file
    Parameters
    ----------
    pdb_list_yaml

    Returns
    -------

    """
    import yaml

    print(f"Loading pdb list from {pdb_list_yaml}...")
    with open(pdb_list_yaml, "r") as f:
        pdb_dict = yaml.safe_load(f)
    return pdb_dict


def download_pdb_structure(pdb_id: str, directory: str):
    """
    Download a PDB structure. If the structure is not available in PDB format, it will be download
    in CIF format.

    Copied with some changes from kinoml.databases.pdb.

    Parameters
    ----------
    pdb_id: str
        The PDB ID of interest.
    directory: str or Path, default=user_cache_dir
        The directory for saving the downloaded structure.

    Returns
    -------
    : Path or False
        The path to the the downloaded file if successful, else False.
    """
    from .utils import download_file
    import os

    # check for structure in PDB format
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_path = os.path.join(directory, f"rcsb_{pdb_id.upper()}.pdb")
    if not os.path.exists(pdb_path):
        print("Downloading PDB entry in PDB format ...")
        if download_file(url, pdb_path):
            return pdb_path
    else:
        return pdb_path

    # check for structure in CIF format
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    cif_path = os.path.join(directory, f"rcsb_{pdb_id.upper()}.cif")
    if not os.path.exists(cif_path):
        print("Downloading PDB entry in CIF format ...")
        if download_file(url, cif_path):
            return cif_path
    else:
        return cif_path
    print(f"Could not download PDB entry {pdb_id}.")
    return False


def download_PDBs(pdb_list, pdb_dir):
    """
    Downloads pdbs from pdb_list_yaml using Kinoml.

    Parameters
    ----------
    pdb_list
    pdb_dir

    Returns
    -------

    """
    # from kinoml.databases.pdb import download_pdb_structure
    from ..data.utils import download_file
    import os

    if not os.path.exists(pdb_dir):
        os.mkdir(pdb_dir)

    print(f"Downloading PDBs to {pdb_dir}")
    for pdb in pdb_list:
        print(pdb)
        download_pdb_structure(pdb, pdb_dir)


def pymol_alignment(
    pdb_path,
    ref_path,
    out_path,
    sel_dict=None,
    mobile_chain_id="A",
    ref_chain_id="A",
):
    """
    Uses Pymol to align a pdb to reference and save the aligned file.
    Can use a dictionary of the form {'name': 'pymol selection string'} to save different selections.
    Parameters
    ----------
    pdb_path
    ref_path
    out_path
    sel_dict

    Returns
    -------

    """
    # TODO: convert this so that I can load all pdbs at once and align them all to ref
    # TODO: Do we need to add pymol to our environment yaml file or is this optional?
    import pymol

    pymol.cmd.load(pdb_path, "mobile")
    pymol.cmd.load(ref_path, "ref")
    pymol.cmd.align(
        f"polymer and name CA and mobile and chain {mobile_chain_id}",
        f"polymer and name CA and ref and chain {ref_chain_id}",
        quiet=0,
    )
    pymol.cmd.save(out_path, "mobile")

    if sel_dict:
        for name, selection in sel_dict.items():
            # get everything but the '.pdb' suffix and then add the name
            sel_path = f"{out_path.split('.')[0]}_{name}.pdb"
            print(f"Saving selection '{selection}' to {sel_path}")
            pymol.cmd.save(sel_path, f"mobile and {selection}")
    pymol.cmd.delete("all")


def align_all_pdbs(
    pdb_list, pdb_dir_path, ref_path=None, ref_name=None, sel_dict=None
):
    """
    Given a list of PDB_IDs and the directory to them, align all to a ref or to the first in the list.
    Parameters
    ----------
    pdb_list
    pdb_dir_path
    ref_path
    ref_name
    sel_dict

    Returns
    -------

    """
    import os

    if not ref_path:
        # Use the first pdb in the list as the reference
        ref = pdb_list[0]
        ref_path = os.path.join(pdb_dir_path, f"rcsb_{ref}.pdb")
    else:
        ref = ref_name
    for pdb in pdb_list:
        pdb_path = os.path.join(pdb_dir_path, f"rcsb_{pdb}.pdb")
        new_pdb_path = os.path.join(pdb_dir_path, f"{pdb}_aligned_to_{ref}.pdb")
        print(
            f"Aligning {pdb_path} \n"
            f"to {ref_path} \n"
            f"and saving to {new_pdb_path}"
        )
        pymol_alignment(pdb_path, ref_path, new_pdb_path, sel_dict)
