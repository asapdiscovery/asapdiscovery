VAULT_URL = "https://app.collaborativedrug.com/api/v1/vaults/5549/"
## All molecules with SMILES (public)
ALL_SMI_SEARCH = "searches/9469227-zd2doWwzJ63bZYaI_vkjXg"
## Noncovalent molecules with experimental measurements (from John)
NONCOVALENT_SMI_SEARCH = "searches/9737468-RPSZ3XnVP-ufU6nNTJjZ_Q"

import logging


def download_url(search_url, header, timeout=5000, retry_delay=5):
    """
    Make requests to the API using the passed information.

    Parameters
    ----------
    search_url : string
        URL for the initial GET request
    header : dict
        Header information passed to GET request. Must contain an entry for
        'X-CDD-token' that gives the user's CDD API token
    timeout : int, default=5000
        Timeout (in seconds)
    retry_delay : int, default=5
        Delay between retry status (in seconds)

    Returns
    -------
    requests.Response
        Response object from the final export GET request
    """
    import requests
    import sys
    import time

    ## Make the initial download request
    logging.debug(f"download_url : initiating search {search_url}")
    response = requests.get(search_url, headers=header)
    logging.debug(f"  {response}")
    export_id = response.json()["id"]
    logging.debug(f"  Export id for requested search is {export_id}")

    ## Check every `retry_delay` seconds to see if the export is ready
    status_url = f"{VAULT_URL}export_progress/{export_id}"
    status = None
    total_seconds = 0
    while True:
        logging.debug(f"  checking if export is finished at {status_url}")
        response = requests.get(status_url, headers=header)
        status = response.json()["status"]

        if status == "finished":
            logging.debug(f"  Export is ready")
            break

        ## Sleep between attempts
        time.sleep(retry_delay)
        total_seconds += retry_delay

        ## Time out when we reach the limit
        if total_seconds > timeout:
            logging.error("Export Never Finished")
            break

    if status != "finished":
        logging.error(
            f"CDD Vault export timed out. Please check manually: {search_url}"
        )
        sys.exit("Export failed")

    ## Send GET request for final export
    result_url = f"{VAULT_URL}exports/{export_id}"
    response = requests.get(result_url, headers=header)

    return response


def download_achiral(header, fn_out=None):
    """
    Download all molecules and remove any chiral molecules.

    Parameters
    ----------
    header : dict
        Header information passed to GET request. Must contain an entry for
        'X-CDD-token' that gives the user's CDD API token
    fn_out : str, optional
        CSV to save compound information to

    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all achiral molecules
    """
    import pandas
    from io import StringIO

    from .utils import get_achiral_molecules

    ## Download all molecules to start
    response = download_url(VAULT_URL + NONCOVALENT_SMI_SEARCH, header)
    ## Parse into DF
    mol_df = pandas.read_csv(StringIO(response.content.decode()))
    ## Get rid of any molecules that snuck through without SMILES
    idx = (
        mol_df.loc[:, ["shipment_SMILES", "suspected_SMILES"]]
        .isna()
        .all(axis=1)
    )
    mol_df = mol_df.loc[~idx, :].copy()
    ## Some of the SMILES from CDD have extra info at the end
    mol_df.loc[:, "shipment_SMILES"] = [
        s.strip("|").split()[0] if not pandas.isna(s) else s
        for s in mol_df.loc[:, "shipment_SMILES"]
    ]
    mol_df.loc[:, "suspected_SMILES"] = [
        s.strip("|").split()[0] if not pandas.isna(s) else s
        for s in mol_df.loc[:, "suspected_SMILES"]
    ]

    ## Remove chiral molecules
    achiral_df = get_achiral_molecules(mol_df)

    ## Save to CSV as requested
    if fn_out:
        achiral_df.to_csv(fn_out, index=False)

    return achiral_df
