import os
from zipfile import ZipFile

import requests

BASE_URL = "https://fragalysis.diamond.ac.uk/api/download_structures/"
# Info for the POST call
MPRO_API_CALL = {
    "target_name": "Mpro",
    "proteins": "",
    "event_info": "false",
    "sigmaa_info": "false",
    "diff_info": "false",
    "trans_matrix_info": "false",
    "NAN": "false",
    "mtz_info": "false",
    "cif_info": "false",
    "NAN2": "false",
    "map_info": "false",
    "single_sdf_file": "false",
    "sdf_info": "true",
    "pdb_info": "true",
    "bound_info": "true",
    "metadata_info": "true",
    "smiles_info": "true",
    "static_link": "false",
    "file_url": "",
}


def download(out_fn, extract=True):
    """
    Download Mpro structures from fragalysis.

    Parameters
    ----------
    out_fn : str
        Where to save the downloaded zip file
    extract : bool, default=True
        Whether to extract the zip file after downloading. Extracts to the
        directory given by `dirname(out_fn)`
    """
    # First send POST request to prepare the download file and get its URL
    r = requests.post(BASE_URL, data=MPRO_API_CALL)
    url_dl = r.text.split(':"')[1].strip('"}')
    print("Downloading archive", flush=True)
    # Send GET request for the zip archive
    r_dl = requests.get(BASE_URL, params={"file_url": url_dl})
    # Full archive stored in r_dl.content, so write to zip file
    with open(out_fn, "wb") as fp:
        fp.write(r_dl.content)

    # Extract files if requested
    if extract:
        print("Extracting files", flush=True)
        zf = ZipFile(out_fn)
        zf.extractall(path=os.path.dirname(out_fn))


def parse_xtal(x_fn, x_dir, p_only=True):
    """
    Load all crystal structures into schema.CrystalCompoundData objects.
    Parameters
    ----------
    x_fn : str
        CSV file giving information on each crystal structure
    x_dir : str
        Path to directory containing directories with crystal structure PDB
        files
    p_only : bool, default=True
        Whether to filter to only include fragalysis structures of the
        format Mpro-P*
    Returns
    -------
    List[schema.CrystalCompoundData]
        List of parsed crystal structures
    """
    import pandas

    from .schema import CrystalCompoundData

    df = pandas.read_csv(x_fn)

    if p_only:
        # Find all P-files
        idx = [(type(d) is str) and ("-P" in d) for d in df["Dataset"]]
    else:
        idx = [type(d) is str for d in df["Dataset"]]

    # Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [
        dict(zip(("smiles", "dataset", "compound_id"), r[1].values))
        for r in df.loc[idx, ["SMILES", "Dataset", "Compound ID"]].iterrows()
    ]

    # Add structure filename information
    for d in xtal_dicts:
        fn_base = f'{x_dir}/{d["dataset"]}_0{{}}/{d["dataset"]}_0{{}}_{{}}.pdb'
        for suf in ["seqres", "bound"]:
            for chain in ["A", "B"]:
                fn = fn_base.format(chain, chain, suf)
                if os.path.isfile(fn):
                    d["str_fn"] = fn
                    break
            if os.path.isfile(fn):
                break
        assert os.path.isfile(fn), f'No structure found for {d["dataset"]}.'

    # Build CrystalCompoundData objects for each row
    xtal_compounds = [CrystalCompoundData(**d) for d in xtal_dicts]

    return xtal_compounds
