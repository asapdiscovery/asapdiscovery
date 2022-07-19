from io import StringIO
import pandas
import requests
import sys
import time

from .utils import get_achiral_molecules

BASE_URL = 'https://app.collaborativedrug.com/api/v1/vaults/5549/'
## All molecules with SMILES (public)
ALL_SMI_SEARCH = 'searches/8975987-kmJ-vR0fhkdccPw5UdWiIA'

def download_url(url, header):
    """
    Make requests to the API using the passed information.

    Parameters
    ----------
    url : string
        URL for the initial GET request
    header : dict
        Header information passed to GET request. Must contain an entry for
        'X-CDD-token' that gives the user's CDD API token

    Returns
    -------
    requests.Response
        Response object from the final export GET request
    """
    ## Make the initial download request
    response = requests.get(url, headers=header)
    export_id = response.json()['id']
    url = f'{BASE_URL}export_progress/{export_id}'

    ## Check every 5 seconds to see if the export is ready
    status = None
    total_seconds = 0
    while status != 'finished':
        response = requests.get(url, headers=header)
        status = response.json()['status']

        time.sleep(5)
        total_seconds += 5
        ## Time out after 5000 seconds
        if total_seconds > 5000:
            print('Export Never Finished')
            break

    if status != 'finished':
        sys.exit('EXPORT IS BROKEN')

    ## Send GET request for final export
    url = f'{BASE_URL}exports/{export_id}'
    response = requests.get(url, headers=header)

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
    response = download_url(BASE_URL+ALL_SMI_SEARCH, header)
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
