import os
from pathlib import Path
import requests
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

    # Add structure filename information and filter if not found
    filtered_xtal_dicts = []
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
        if os.path.isfile(fn):
            filtered_xtal_dicts.append(d)
        else:
            print(f'No structure found for {d["dataset"]}.')
    assert (
        len(filtered_xtal_dicts) > 0
    ), "No structure filenames were found by parse_xtal"
    # Build CrystalCompoundData objects for each row
    print(f"Loading {len(filtered_xtal_dicts)} structures")
    xtal_compounds = [CrystalCompoundData(**d) for d in filtered_xtal_dicts]

    return xtal_compounds


def parse_fragalysis(
    x_fn,
    x_dir,
    name_filter=None,
    name_filter_column="crystal_name",
    drop_duplicate_datasets=False,
):
    """
    Load all crystal structures into schema.CrystalCompoundData objects.
    Parameters
    ----------
    x_fn : str
        metadata.CSV file giving information on each crystal structure
    x_dir : str
        Path to directory containing directories with crystal structure PDB
        files
    name_filter : str or list
        String or list of strings that are required to be in the name_filter_column
    name_filter_column : str
        Name of column in the metadata.csv that will be used to filter the dataframe
    drop_duplicate_datasets : bool
        If true, will drop the _1A, _0B, etc duplicate datasets for a given crystal structure.
    Returns
    -------
    List[schema.CrystalCompoundData]
        List of parsed crystal structures
    """
    import pandas
    from .schema import CrystalCompoundData
    from tqdm import tqdm

    x_dir = Path(x_dir)

    df = pandas.read_csv(x_fn)

    # Only keep rows of dataframe where the name_filter_column includes the name_filter string
    if name_filter:
        if type(name_filter) == str:
            idx = df[name_filter_column].apply(lambda x: name_filter in x)
            df = df[idx]
        elif type(name_filter) == list:
            for filter in name_filter:
                idx = df[name_filter_column].apply(lambda x: filter in x)
                df = df[idx]
    # Drop duplicates, keeping only the first one.
    if drop_duplicate_datasets:
        df = df.drop_duplicates("RealCrystalName")

    # Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [
        dict(zip(("smiles", "dataset", "compound_id"), r[1].values))
        for r in df.loc[
            :, ["smiles", "crystal_name", "alternate_name"]
        ].iterrows()
    ]

    # Add structure filename information and filter if not found
    filtered_xtal_dicts = []
    for d in tqdm(xtal_dicts):
        glob_str = f"{d['dataset']}*/*.pdb"
        fns = list(x_dir.glob(glob_str))
        for fn in fns:
            d["str_fn"] = str(fn)

            # This should basically always be true since we're getting the filenames from glob but just in case.
            if os.path.isfile(fn):
                filtered_xtal_dicts.append(d)
    assert (
        len(filtered_xtal_dicts) > 0
    ), "No structure filenames were found by parse_fragalysis"

    # Build CrystalCompoundData objects for each row
    print(f"Loading {len(filtered_xtal_dicts)} structures")
    xtal_compounds = [CrystalCompoundData(**d) for d in filtered_xtal_dicts]
        assert os.path.isfile(fn), f'No structure found for {d["dataset"]}.'

    return xtal_compounds
