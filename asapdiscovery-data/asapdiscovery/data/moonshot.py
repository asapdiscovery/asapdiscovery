import logging
from io import StringIO

import pandas

# Base CDD vault API URL
CDD_URL = "https://app.collaborativedrug.com/api/v1/vaults"
# Vault number for the Moonshot vault
MOONSHOT_VAULT = "5549"
# All molecules with SMILES (public)
ALL_SMI_SEARCH = "9469227-zd2doWwzJ63bZYaI_vkjXg"
# Noncovalent molecules with experimental measurements (from John)
NONCOVALENT_SMI_SEARCH = "9737468-RPSZ3XnVP-ufU6nNTJjZ_Q"
# Noncovalent with experimental measurements, including batch created date
NONCOVALENT_W_DATES_SEARCH = "11947939-KXLWU3JLbLzI354es-VKVg"


def download_url(search_url, header, vault=None, timeout=5000, retry_delay=5):
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
    import sys
    import time

    import requests

    # If vault is not specified, attempt to parse from URL
    if not vault:
        vault = search_url.split("/")[-3]
        logging.debug(f"Using {vault} as vault.")

    # Make the initial download request
    logging.debug(f"download_url : initiating search {search_url}")
    response = requests.get(search_url, headers=header)
    logging.debug(f"  {response}")
    export_id = response.json()["id"]
    logging.debug(f"  Export id for requested search is {export_id}")

    # Check every `retry_delay` seconds to see if the export is ready
    status_url = f"{CDD_URL}/{vault}/export_progress/{export_id}"
    status = None
    total_seconds = 0
    while True:
        logging.debug(f"  checking if export is finished at {status_url}")
        response = requests.get(status_url, headers=header)
        status = response.json()["status"]

        if status == "finished":
            logging.debug("  Export is ready")
            break

        # Sleep between attempts
        time.sleep(retry_delay)
        total_seconds += retry_delay

        # Time out when we reach the limit
        if total_seconds > timeout:
            logging.error("Export Never Finished")
            break

    if status != "finished":
        logging.error(
            f"CDD Vault export timed out. Please check manually: {search_url}"
        )
        sys.exit("Export failed")

    # Send GET request for final export
    result_url = f"{CDD_URL}/{vault}/exports/{export_id}"
    response = requests.get(result_url, headers=header)

    return response


# TODO: Generalize inclusion criteria to something more compact
def download_molecules(
    header,
    smiles_fieldname="suspected_SMILES",
    fn_out=None,
    fn_cache=None,
    **filter_kwargs,
):
    """
    Download all molecules and filter based on args in `filter_kwargs`. Saves
    and loads unfiltered CSV file to `fn_cache` if provided, and saves filtered
    CSV file to `fn_out` if provided.

    Parameters
    ----------
    header : dict
        Header information passed to GET request. Must contain an entry for
        'X-CDD-token' that gives the user's CDD API token
    smiles_fieldname : str, default='suspected_SMILES'
        Field to use to extract SMILES
    fn_out : str, optional
        If specified, filename to write CSV to
    fn_cache : str, optional
        If specified, file to write unfiltered CSV download to
    filter_kwargs :
        Other arguments passed to filter_molecules

    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all achiral molecules
    """
    import os

    if fn_cache and os.path.exists(fn_cache):
        with open(fn_cache) as infile:
            content = infile.read()
    else:
        # Download all molecules to start
        url = f"{CDD_URL}/{MOONSHOT_VAULT}/searches/{NONCOVALENT_W_DATES_SEARCH}"
        logging.debug(f"Downloading data from CDD vault from {url}")
        response = download_url(url, header, vault=MOONSHOT_VAULT)
        content = response.content.decode()

        if fn_cache:
            with open(fn_cache, "w") as outfile:
                outfile.write(content)

    # Parse into DF
    mol_df = pandas.read_csv(StringIO(content))
    logging.debug(f"\n{mol_df}")

    # Remove chiral molecules
    logging.debug("Filtering dataframe...")
    from .utils import filter_molecules_dataframe

    filtered_df = filter_molecules_dataframe(mol_df, **filter_kwargs)

    # Save to CSV as requested
    if fn_out:
        logging.debug(f"Generating CSV file {fn_out}")
        filtered_df.to_csv(fn_out, index=False)

    return filtered_df
