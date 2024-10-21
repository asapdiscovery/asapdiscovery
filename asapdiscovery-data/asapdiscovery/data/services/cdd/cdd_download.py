import logging
from io import StringIO

import pandas

# Base CDD vault API URL
CDD_URL = "https://app.collaborativedrug.com/api/v1/vaults"
# All molecules with SMILES (public)
MOONSHOT_ALL_SMI_SEARCH = "13157856-vbatz0uAL8fhJR87pFN0tA"
# Noncovalent molecules with experimental measurements (from John)
MOONSHOT_NONCOVALENT_SMI_SEARCH = "9737468-RPSZ3XnVP-ufU6nNTJjZ_Q"
# Noncovalent with experimental measurements, including batch created date
MOONSHOT_NONCOVALENT_W_DATES_SEARCH = "11947939-KXLWU3JLbLzI354es-VKVg"

MOONSHOT_SEARCH_DICT = {
    "sars_fluorescence_all_smi": MOONSHOT_ALL_SMI_SEARCH,
    "sars_fluorescence_noncovalent_no_dates": MOONSHOT_NONCOVALENT_SMI_SEARCH,
    "sars_fluorescence_noncovalent_w_dates": MOONSHOT_NONCOVALENT_W_DATES_SEARCH,
}

# All molecules with Mac1 FRET data
ASAP_MAC1_ALL_FRET = "13002158-OsTakM3U--QoAEusMICUDA"


def download_url(search_url, header, vault=None, timeout=5000, retry_delay=10):
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
    retry_delay : int, default=10
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
    vault=None,
    search="sars_fluorescence_noncovalent_w_dates",
    fn_out=None,
    fn_cache=None,
    **kwargs,
):
    """
    Download all molecules and filter based on args in `kwargs`. Saves
    and loads unfiltered CSV file to `fn_cache` if provided, and saves filtered
    CSV file to `fn_out` if provided.

    Parameters
    ----------
    header : dict
        Header information passed to GET request. Must contain an entry for
        'X-CDD-token' that gives the user's CDD API token
    vault : str, default=None
        Which CDD vault to search through. By default use the Moonshot vault
    search : str, default="sars_fluorescence_noncovalent_w_dates"
        Which entry in MOONSHOT_SEARCH_DICT to use as the search id. If the given value
        can't be found, assume it's the actual search id and try to download
    fn_out : str, optional
        If specified, filename to write CSV to
    fn_cache : str, optional
        If specified, file to write unfiltered CSV download to
    kwargs : dict
        Other arguments passed to filter_molecules_dataframe and
        parse_fluorescence_data_cdd

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
        if not vault:
            try:
                vault = os.environ["MOONSHOT_CDD_VAULT_NUMBER"]
            except KeyError:
                raise ValueError("No value specified for vault.")
        # First try and get the search id from our known searches, otherwise assume the
        #  given value is the search id itself
        try:
            search_id = MOONSHOT_SEARCH_DICT[search]
        except KeyError:
            logging.debug(f"Using {search} as the search id directly.")
            search_id = search
        url = f"{CDD_URL}/{vault}/searches/{search_id}"
        logging.debug(f"Downloading data from CDD vault from {url}")
        response = download_url(url, header, vault=vault)
        content = response.content.decode()

        if fn_cache:
            with open(fn_cache, "w") as outfile:
                outfile.write(content)

    # Parse into DF
    mol_df = pandas.read_csv(StringIO(content))
    logging.debug(f"\n{mol_df}")

    # Remove chiral molecules
    logging.debug("Filtering dataframe...")
    from asapdiscovery.data.util.utils import (
        filter_molecules_dataframe,
        parse_fluorescence_data_cdd,
    )

    filter_kwargs = [
        "id_fieldname",
        "smiles_fieldname",
        "assay_name",
        "retain_achiral",
        "retain_racemic",
        "retain_enantiopure",
        "retain_semiquantitative_data",
    ]
    filter_kwargs = {k: kwargs[k] for k in filter_kwargs if k in kwargs}
    filtered_df = filter_molecules_dataframe(mol_df, **filter_kwargs)
    parse_kwargs = [
        "keep_best_per_mol",
        "assay_name",
        "dG_T",
        "cp_values",
        "pic50_stderr_filt",
    ]
    parse_kwargs = {k: kwargs[k] for k in parse_kwargs if k in kwargs}
    parsed_df = parse_fluorescence_data_cdd(filtered_df, **parse_kwargs)

    # Save to CSV as requested
    if fn_out:
        logging.debug(f"Generating CSV file {fn_out}")
        parsed_df.to_csv(fn_out, index=False)

    return parsed_df
