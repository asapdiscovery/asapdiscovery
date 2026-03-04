import os
from pathlib import Path
from zipfile import ZipFile

import requests
from asapdiscovery.data.schema.legacy import CrystalCompoundData
from asapdiscovery.data.util.stringenum import StringEnum

BASE_URL = "https://fragalysis.diamond.ac.uk/api/download_structures/"
BASE_URL_LEGACY = (
    "https://fragalysis-legacy.xchem.diamond.ac.uk/api/download_structures/"
)

# Info for the POST call
API_CALL_BASE = {
    "target_name": "",
    "proteins": "",
    "event_info": False,
    "sigmaa_info": False,
    "diff_info": False,
    "trans_matrix_info": False,
    "NAN": False,
    "mtz_info": False,
    "cif_info": False,
    "NAN2": False,
    "map_info": False,
    "single_sdf_file": True,
    "sdf_info": True,
    "pdb_info": False,
    "bound_info": True,
    "metadata_info": True,
    "smiles_info": True,
    "static_link": False,
    "file_url": "",
}


API_CALL_BASE_LEGACY = {
    "target_name": "",
    "file_url": "",
    "proteins": "",
}


class FragalysisTargets(StringEnum):
    SARS = "Mpro"
    MAC1 = "Mac1"
    D68EV3CPRO = "D68EV3CPROA"
    NPROT = "Nprot"
    nsp13 = "nsp13"
    XX01ZVNS2B = "XX01ZVNS2B"


def download(out_fn, api_call, extract=True, base_url=BASE_URL):
    """
    Download target structures from fragalysis.

    Parameters
    ----------
    out_fn : Union[str, Path]
        Where to save the downloaded zip file
    api_call : dict
        Dictionary containing args for the POST request. Target is specified here.
    extract : bool, default=True
        Whether to extract the zip file after downloading. Extracts to the
        directory given by `dirname(out_fn)`
    """
    # First send POST request to prepare the download file and get its URL
    r = requests.post(base_url, json=api_call)
    if not r.ok:
        raise requests.HTTPError(
            f"Post request to {base_url} failed with {r.status_code} error code, "
            f"using the following API call {api_call}."
        )
    url_dl = r.json()["file_url"]
    print("Downloading archive", flush=True)
    # Send GET request for the zip archive
    r_dl = requests.get(base_url, params={"file_url": url_dl})
    # Full archive stored in r_dl.content, so write to zip file
    with open(out_fn, "wb") as fp:
        fp.write(r_dl.content)

    # Extract files if requested
    if extract:
        extract_zip(out_fn)


# TODO: move this function to utils or similar, if we end up needing it somewhere else
def extract_zip(out_fn):
    """Extracts contents of zip file

    Parameters
    ----------
    out_fn: str or Path
        Zip file path to extract
    """
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
    x_fn : str or Path
        metadata.CSV file giving information on each crystal structure
    x_dir : str or Path
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
    from tqdm import tqdm

    x_dir = Path(x_dir)

    df = pandas.read_csv(x_fn)

    # Only keep rows of dataframe where the name_filter_column includes the name_filter string
    if name_filter:
        if isinstance(name_filter, str):
            idx = df[name_filter_column].apply(lambda x: name_filter in x)
            df = df[idx]
        elif isinstance(name_filter, list):
            for filter in name_filter:
                idx = df[name_filter_column].apply(lambda x: filter in x)
                df = df[idx]
    # Drop duplicates, keeping only the first one.
    if drop_duplicate_datasets:
        df = df.drop_duplicates("RealCrystalName")

    # Remove whitespace from the the relevant columns
    df["smiles"].str.strip()
    df["crystal_name"].str.strip()
    df["alternate_name"].str.strip()

    # Build argument dicts for the CrystalCompoundData objects
    try:
        xtal_dicts = [
            dict(zip(("smiles", "dataset", "compound_id"), r[1].values))
            for r in df.loc[:, ["smiles", "crystal_name", "alternate_name"]].iterrows()
        ]
    except KeyError as e:
        raise Exception(
            "Did you use 'Mpro_compound_tracker_csv.csv'? Use 'metadata.csv' instead. "
            "This CSV is expected to contain columns 'smiles', 'crystal_name', and 'alternate_name', which correspond "
            "to the SD tags 'smiles', 'dataset', and 'compound_id' respectively."
        ) from e

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
