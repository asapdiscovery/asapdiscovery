import argparse
from io import StringIO
import os
import pandas
from rdkit.Chem import FindMolChiralCenters, MolFromSmiles
import requests
import sys
import time

BASE_URL = 'https://app.collaborativedrug.com/api/v1/vaults/5549/'
## All molecules with SMILES (public)
ALL_SMI_SEARCH = 'searches/8975987-kmJ-vR0fhkdccPw5UdWiIA'

def download(url, header):
    response = requests.get(url, headers=header)
    export_id = response.json()['id']
    url = f'{BASE_URL}export_progress/{export_id}'

    status = None
    total_seconds = 0
    while status != 'finished':
        response = requests.get(url, headers=header)
        status = response.json()['status']

        time.sleep(5)
        total_seconds += 5
        if total_seconds > 5000:
            print('Export Never Finished')
            break

    if status != 'finished':
        sys.exit('EXPORT IS BROKEN')

    url = f'{BASE_URL}exports/{export_id}'
    response = requests.get(url, headers=header)

    return(response)

def download_achiral(header, fn_out=None):
    response = download(BASE_URL+ALL_SMI_SEARCH, header)
    mol_df = pandas.read_csv(StringIO(response.content.decode()))
    ## Get rid of any molecules that snuck through without SMILES
    idx = mol_df.loc[:,['shipment_SMILES', 'suspected_SMILES']].isna().all(axis=1)
    mol_df = mol_df.loc[~idx,:].copy()
    ## Some of the SMILES from CDD have extra info at the end
    mol_df.loc[:,'shipment_SMILES'] = [s.strip('|').split()[0] \
        for s in mol_df.loc[:,'shipment_SMILES']]
    mol_df.loc[:,'suspected_SMILES'] = [s.strip('|').split()[0] \
        for s in mol_df.loc[:,'suspected_SMILES']]

    achiral_df = get_achiral_molecules(mol_df)

    if fn_out:
        achiral_df.to_csv(fn_out, index=False)

    return(achiral_df)

def get_achiral_molecules(mol_df):
    ## Check whether a SMILES is chiral or not
    check_achiral = lambda smi: len(FindMolChiralCenters(MolFromSmiles(smi),
        includeUnassigned=True, includeCIP=False,
        useLegacyImplementation=False)) == 0
    achiral_idx = []
    for _, r in mol_df.iterrows():
        if not pandas.isna(r['suspected_SMILES']):
            achiral_idx.append(check_achiral(r['suspected_SMILES']))
        elif not pandas.isna(r['shipment_SMILES']):
            achiral_idx.append(check_achiral(r['shipment_SMILES']))
        else:
            raise ValueError(f'No SMILES found for {r["Canonical PostEra ID"]}')

    return(mol_df.loc[achiral_idx,:])

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-tok', required=True,
        help='File containing CDD token.')
    parser.add_argument('-o', required=True, help='Output CSV file.')

    return(parser.parse_args())

def main():
    args = get_args()

    ## All molecules with SMILES (public)
    search_id = 'searches/8975987-kmJ-vR0fhkdccPw5UdWiIA'
    ## All molecules with SMILES
    # search_id = 'searches/8869579-9SKN2Zs9LaTRLq9PmTQzZg'
    ## Protease assay
    # search_id = 'protocols/49439/data?async=true'
    ## Achiral in stereochem comments
    # search_id = 'searches/8866255-iPG5Uqf4mGDemE65iuWt0w'
    ## Test search (pair of enantiomers)
    # search_id = 'searches/8860834-q4Jf0BkQNln6mDCG6xtp3w'
    url = f'https://app.collaborativedrug.com/api/v1/vaults/5549/{search_id}'
    header = {'X-CDD-token': ''.join(open(args.tok, 'r').readlines()).strip()}

    response = download(url, header)
    mol_df = pandas.read_csv(StringIO(response.content.decode()))
    ## Get rid of any molecules that snuck through without SMILES
    idx = mol_df.loc[:,['shipment_SMILES', 'suspected_SMILES']].isna().all(axis=1)
    mol_df = mol_df.loc[~idx,:].copy()
    ## Some of the SMILES from CDD have extra info at the end
    mol_df.loc[:,'shipment_SMILES'] = [s.strip('|').split()[0] \
        for s in mol_df.loc[:,'shipment_SMILES']]
    mol_df.loc[:,'suspected_SMILES'] = [s.strip('|').split()[0] \
        for s in mol_df.loc[:,'suspected_SMILES']]

    achiral_df = get_achiral_molecules(mol_df)
    achiral_df.to_csv(args.o, index=False)

if __name__ == '__main__':
    main()