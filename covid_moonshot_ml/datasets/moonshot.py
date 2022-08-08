from io import StringIO
import pandas
import requests
import sys
import time

# Handle logging
import logging

from .utils import get_achiral_molecules

VAULT_URL = 'https://app.collaborativedrug.com/api/v1/vaults/5549/'
## All molecules with SMILES (public)
# This string comes from accessing a saved CDD search and copying the trailing `searches/...` part of the URL
ALL_SMI_SEARCH = 'searches/8975987-kmJ-vR0fhkdccPw5UdWiIA'
NONCOVALENT_SMI_SEARCH = 'searches/9737468-RPSZ3XnVP-ufU6nNTJjZ_Q'

def download_url(search_url, header, timeout=5000, retry_delay=5):
    """
    Make requests to the API using the passed information.

    Parameters
    ----------
    search_url : string
        URL for the initial search GET request
    header : dict
        Header information passed to GET request. Must contain an entry for
        'X-CDD-token' that gives the user's CDD API token
    timeout : float, optional, default=5000
        Timeout (in seconds)
    retry_delay : float, optional, default=5
        Delay between retry status (in seconds)

    Returns
    -------
    requests.Response
        Response object from the final export GET request
    """
    ## Make the initial download request
    logging.debug(f'download_url : initiating search {search_url}')
    response = requests.get(search_url, headers=header)
    logging.debug(f'  {response}')
    export_id = response.json()['id']
    logging.debug(f'  Export id for requested search is {export_id}')

    ## Check every 5 seconds to see if the export is ready
    status_url = f'{VAULT_URL}export_progress/{export_id}'
    status = None
    total_seconds = 0
    while True:
        logging.debug(f'  checking if export is finished at {status_url}')
        response = requests.get(status_url, headers=header)
        status = response.json()['status']

        if (status == 'finished'):
            logging.debug(f'  Export is ready')
            break

        # Sleep between retries
        time.sleep(retry_delay)
        total_seconds += retry_delay

        # Check if we have reached the timeout
        if total_seconds > timeout:
            logging.error('Export Never Finished')
            break

    if status != 'finished':
        logging.error('CDD Vault export timed out. Please check manually: {search_url}')
        sys.exit('Export failed')

    ## Send GET request for final export
    result_url = f'{VAULT_URL}exports/{export_id}'
    response = requests.get(result_url, headers=header)

    return(response)

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
    ## Download all molecules to start
    response = download_url(VAULT_URL+NONCOVALENT_SMI_SEARCH, header)
    ## Parse into DF
    mol_df = pandas.read_csv(StringIO(response.content.decode()))
    ## Get rid of any molecules that snuck through without SMILES
    idx = mol_df.loc[:,['shipment_SMILES', 'suspected_SMILES']].isna().all(axis=1)
    mol_df = mol_df.loc[~idx,:].copy()
    ## Some of the SMILES from CDD have extra info at the end
    mol_df.loc[:,'shipment_SMILES'] = [s.strip('|').split()[0] \
        for s in mol_df.loc[:,'shipment_SMILES']]
    mol_df.loc[:,'suspected_SMILES'] = [s.strip('|').split()[0] \
        for s in mol_df.loc[:,'suspected_SMILES']]

    ## Remove chiral molecules
    achiral_df = get_achiral_molecules(mol_df)

    ## Save to CSV as requested
    if fn_out:
        achiral_df.to_csv(fn_out, index=False)

    return(achiral_df)

# TODO: Generalize inclusion criteria to something more compact
def download_molecules(header,
                       smiles_fieldname='suspected_SMILES',
                       fn_out=None,
                       **filter_kwargs):
    """
    Download all molecules and remove any chiral molecules.

    Parameters
    ----------
    header : dict
        Header information passed to GET request. Must contain an entry for
        'X-CDD-token' that gives the user's CDD API token
    smiles_fieldname : str, optional, default='suspected_SMILES'
        Field to use to extract SMILES
    fn_out : str, optional, default=None
        If specified, filename to write CSV to
    filter_kwargs : 
        Other arguments passed to filter_molecules

    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all achiral molecules
    """
    ## DEBUG : use local copy if present
    import os
    if os.path.exists('download.csv'):
        with open('download.csv', 'rt') as infile:
            content = infile.read()
    else:    
        ## Download all molecules to start
        url = VAULT_URL+NONCOVALENT_SMI_SEARCH
        logging.debug(f'Downloading data from CDD vault from {url}')
        response = download_url(url, header)
        content = response.content.decode()

        # DEBUG
        with open('download.csv', 'wt') as outfile:
            outfile.write(content)

    ## Parse into DF
    mol_df = pandas.read_csv(StringIO(content))
    logging.debug(f'\n{mol_df}')

    ## Remove chiral molecules
    logging.debug(f'Filtering dataframe...')
    from .utils import filter_molecules_dataframe
    filtered_df = filter_molecules_dataframe(mol_df, **filter_kwargs)

    ## Save to CSV as requested
    if fn_out:
        logging.debug(f'Generating CSV file {fn_out}')
        filtered_df.to_csv(fn_out, index=False)

    return(filtered_df)
