import os

import pandas
import requests
from zipfile import ZipFile

from asap_dataset_handling.schema import CrystalCompoundData

BASE_URL = "https://fragalysis.diamond.ac.uk/api/download_structures/"
## Info for the POST call
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
    ## First send POST request to prepare the download file and get its URL
    r = requests.post(BASE_URL, data=MPRO_API_CALL)
    url_dl = r.text.split(':"')[1].strip('"}')
    print("Downloading archive", flush=True)
    ## Send GET request for the zip archive
    r_dl = requests.get(BASE_URL, params={"file_url": url_dl})
    ## Full archive stored in r_dl.content, so write to zip file
    with open(out_fn, "wb") as fp:
        fp.write(r_dl.content)

    ## Extract files if requested
    if extract:
        print("Extracting files", flush=True)
        zf = ZipFile(out_fn)
        zf.extractall(path=os.path.dirname(out_fn))


def parse_fragalysis_data(frag_fn, x_dir, cmpd_ids=None, o_dir=False):
    ## Load in csv
    sars2_structures = pandas.read_csv(frag_fn).fillna("")

    if cmpd_ids is not None:
        ## Filter fragalysis dataset by the compounds we want to test
        sars2_filtered = sars2_structures[
            sars2_structures["Compound ID"].isin(cmpd_ids)
        ]
    else:
        sars2_filtered = sars2_structures

    if o_dir:
        mols_wo_sars2_xtal = sars2_filtered[sars2_filtered["Dataset"].isna()][
            ["Compound ID", "SMILES", "Dataset"]
        ]
        mols_w_sars2_xtal = sars2_filtered[~sars2_filtered["Dataset"].isna()][
            ["Compound ID", "SMILES", "Dataset"]
        ]

        ## Use utils function to get sdf file from dataset
        mols_w_sars2_xtal["SDF"] = mols_w_sars2_xtal["Dataset"].apply(
            get_sdf_fn_from_dataset, fragalysis_dir=x_dir
        )

        ## Save csv files for each dataset
        mols_wo_sars2_xtal.to_csv(
            os.path.join(o_dir, "mers_ligands_without_SARS2_structures.csv"),
            index=False,
        )

        mols_w_sars2_xtal.to_csv(
            os.path.join(o_dir, "mers_ligands_with_SARS2_structures.csv"),
            index=False,
        )

    ## Construct sars_xtal list
    sars_xtals = {}
    for data in sars2_filtered.to_dict("index").values():
        cmpd_id = data["Compound ID"]
        dataset = data["Dataset"]
        if len(dataset) > 0:
            ## TODO: is this the behaviour we want? this will build an empty object if there isn't a dataset
            if not sars_xtals.get(cmpd_id) or "-P" in dataset:
                sars_xtals[cmpd_id] = CrystalCompoundData(
                    smiles=data["SMILES"],
                    compound_id=cmpd_id,
                    dataset=dataset,
                    sdf_fn=get_sdf_fn_from_dataset(dataset, x_dir),
                )
        else:
            sars_xtals[cmpd_id] = CrystalCompoundData()

    return sars_xtals


def get_sdf_fn_from_dataset(
    dataset: str,
    fragalysis_dir,
):
    fn = os.path.join(fragalysis_dir, f"{dataset}_0A/{dataset}_0A.sdf")
    if not os.path.exists(fn):
        print(f"File {fn} not found...")
        fn = None  ## not sure what behaviour this should have
    return fn


def get_compound_id_xtal_dicts(sars_xtals):
    """
    Get a pair of dictionaries that map between crystal structures and compound
    ids.

    Parameters
    ----------
    sars_xtals : Iter[CrystalCompoundData]
        Iterable of CrystalCompoundData objects from fragalysis.

    Returns
    -------
    Dict[str: List[str]]
        Dict mapping compound_id to list of crystal structure ids.
    Dict[str: str]
        Dict mapping crystal structure id to compound_id.
    """
    compound_to_xtals = {}
    xtal_to_compound = {}
    for ccd in sars_xtals:
        compound_id = ccd.compound_id
        dataset = ccd.dataset
        try:
            compound_to_xtals[compound_id].append(dataset)
        except KeyError:
            compound_to_xtals[compound_id] = [dataset]

        xtal_to_compound[dataset] = compound_id

    return (compound_to_xtals, xtal_to_compound)
