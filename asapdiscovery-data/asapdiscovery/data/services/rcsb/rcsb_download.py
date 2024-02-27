import os
from typing import Union


def download_pdb_structure(
    pdb_id: str, directory: Union[str, os.PathLike], file_format: str = "pdb"
):
    """
    Download a structure, using the specified format/type.

    Copied with some changes from kinoml.databases.pdb.

    Parameters
    ----------
    pdb_id: str
        The PDB ID of interest.
    directory: str or Path, default=user_cache_dir
        The directory for saving the downloaded structure.
    file_format : str, default="pdb"
        Indicates whether you would like to download the entry in pdb ("pdb")
        or cif format ("cif"), or the first biological assembly in
        cif format ("cif1"). Defaults to "pdb".

    Returns
    -------
    file_path : Path or False
        The path to the downloaded file if successful, else False.
    """
    import os

    import requests
    from asapdiscovery.data.util.utils import download_file

    url_base_str = "https://files.rcsb.org/download/"  # base str to use for URLs
    # Dictionary with allowed formats and their upstream basenames
    format_to_basename = {
        "pdb": f"{pdb_id.lower()}.pdb",
        "cif": f"{pdb_id.lower()}.cif",
        "cif1": f"{pdb_id.lower()}-assembly1.cif",
    }

    allowed_types = format_to_basename.keys()
    # Make sure pdb_type can be handled
    file_format = file_format.lower()
    if file_format not in allowed_types:
        raise NotImplementedError(
            f"pdb_type expected to be one of {allowed_types}, not '{file_format}'"
        )

    basename = format_to_basename[file_format]
    local_path = os.path.join(directory, f"rcsb_{basename}")
    # Download only if it doesn't exist locally
    if not os.path.exists(local_path):
        url = f"{url_base_str}{basename}"
        response = download_file(url, local_path)
        if response.status_code == 200:
            result = local_path
        elif response.ok:
            raise requests.HTTPError(
                f"Received status code {response.status_code}, " "file not downloaded."
            )
        else:
            response.raise_for_status()
    else:
        print(f"{local_path} already exists!...")
        result = local_path

    return result


def download_PDBs(pdb_list, pdb_dir, file_format="pdb", ignore_errors=True):
    """
    Downloads pdbs from pdb_list_yaml using Kinoml.

    Parameters
    ----------
    pdb_list : List[str]
        List of RCSB IDs to download
    pdb_dir : str
        Directory to download structures to
    file_format : str, default="pdb"
        Indicates whether you would like to download the entry in pdb ("pdb")
        or cif format ("cif"), or the first biological assembly in
        cif format ("cif1"). Defaults to "pdb".
    ignore_errors : bool, default=True
        If a PDB file failed to download, either catch the error and ignore, or
        raise the error
    """
    import os

    import requests

    if not os.path.exists(pdb_dir):
        os.mkdir(pdb_dir)

    print(f"Downloading PDBs to {pdb_dir}")
    for pdb in pdb_list:
        print(pdb)
        try:
            download_pdb_structure(pdb, pdb_dir, file_format=file_format)
        except requests.HTTPError as e:
            if ignore_errors:
                print("Error downloading", {pdb}, flush=True)
                continue
            else:
                raise e


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
    Can use a dictionary of the form {'name': 'pymol selection string'}
    to save different selections.

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


def align_all_pdbs(pdb_list, pdb_dir_path, ref_path=None, ref_name=None, sel_dict=None):
    """
    Given a list of PDB_IDs and the directory to them, align all to a ref or to the
    first in the list.

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
