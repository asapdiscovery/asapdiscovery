import argparse
import json
import pandas
from rdkit.Chem import MolToSmiles, MolFromSmiles
import re

from schema import ExperimentalCompoundData, EnantiomerPair, EnantiomerPairList

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-i', required=True, help='CSV file input from CDD.')
    parser.add_argument('-o', required=True, help='Output JSON file.')

    return(parser.parse_args())

def main():
    args = get_args()

    ## Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(args.i)
    df = df.loc[~df['suspected_SMILES'].isna(),:]

    ## Remove stereochemistry tags and get canonical SMILES values (to help
    ##  group stereoisomers)
    smi_nostereo = [MolToSmiles(MolFromSmiles(re.sub('@', '', s))) \
        for s in df['suspected_SMILES']]
    df['suspected_SMILES_nostereo'] = smi_nostereo

    ## Sort by non-stereo SMILES to put the enantiomer pairs together
    df = df.sort_values('suspected_SMILES_nostereo')

    ## Get rid of the </> signs, since we really only need the values to sort
    ##  enantiomer pairs
    pic50_key = 'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50'
    pic50_vals = [float(c[pic50_key].strip('<> ')) for _, c in df.iterrows()]
    df['pIC50'] = pic50_vals

    enant_pairs = []
    ## Loop through the enantiomer pairs and rank them
    for ep in df.groupby('suspected_SMILES_nostereo'):
        ## Make sure there aren't any singletons
        if ep[1].shape[0] != 2:
            print(f'{ep[1].shape[0]} mols for {ep[0]}', flush=True)
            continue

        p = []
        ## Sort by pIC50 value, higher to lower
        ep = ep[1].sort_values('pIC50', ascending=False)
        for _, c in ep.iterrows():
            compound_id = c['Canonical PostEra ID']
            smiles = c['suspected_SMILES']
            experimental_data = {'pIC50': c['pIC50']}

            p.append(ExperimentalCompoundData(
                compound_id=compound_id,
                smiles=smiles,
                racemic=False,
                achiral=False,
                absolute_stereochemistry_enantiomerically_pure=True,
                relative_stereochemistry_enantiomerically_pure=True,
                experimental_data=experimental_data
            ))

        enant_pairs.append(EnantiomerPair(active=p[0], inactive=p[1]))

    ep_list = EnantiomerPairList(pairs=enant_pairs)

    with open(args.o, 'w') as fp:
        fp.write(ep_list.json())

if __name__ == '__main__':
    main()
