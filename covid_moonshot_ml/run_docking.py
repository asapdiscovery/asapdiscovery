import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import pandas
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import rdFMCS

from schema import ExperimentalCompoundDataUpdate, CrystalCompoundData, \
    EnantiomerPairList

from kinoml.features.complexes import OEPositDockingFeaturizer
from kinoml.core.components import BaseProtein
from kinoml.core.ligands import RDKitLigand
from kinoml.core.systems import ProteinLigandComplex

def build_docking_systems(exp_compounds, xtal_compounds, compound_idxs):
    systems = []
    for (c, idx) in zip(exp_compounds, compound_idxs):
        x = xtal_compounds[idx][0]
        protein = BaseProtein(name='MPRO')
        protein.path = x.str_fn
        protein.chain_id = x.str_fn.split('_')[-2][-1]
        protein.expo_id = 'LIG'
        ligand = RDKitLigand.from_smiles(smiles=c.smiles, name=c.compound_id)
        systems.append(ProteinLigandComplex(components=[protein, ligand]))

    return(systems)

def mp_func(exp_mol, search_mols, top_n):
    return(rank_structures(exp_mol, search_mols)[:top_n])

def parse_xtal(x_fn, x_dir):
    df = pandas.read_csv(x_fn)

    ## Find all P-files
    idx = [(type(d) is str) and ('-P' in d) for d in df['Dataset']]

    ## Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [dict(zip(('smiles', 'dataset'), r[1].values)) \
        for r in df.loc[idx,['SMILES', 'Dataset']].iterrows()]

    ## Add structure filename information
    for d in xtal_dicts:
        fn_base = (f'{x_dir}/{d["dataset"]}_0{{}}/{d["dataset"]}_0{{}}_'
            'seqres.pdb')
        fn = fn_base.format('A', 'A')
        if os.path.isfile(fn):
            d['str_fn'] = fn
        else:
            fn = fn_base.format('B', 'B')
            assert os.path.isfile(fn), f'No structure found for {d["dataset"]}.'
            d['str_fn'] = fn

    ## Build CrystalCompoundData objects for each row
    xtal_compounds = np.asarray([CrystalCompoundData(**d) for d in xtal_dicts])

    return(xtal_compounds)

def rank_structures(exp_mol, search_mols):
    match_results = []
    for mol in search_mols:
        ## Perform MCS search for each search molecule
        # maximize atoms first and then bonds
        mcs = rdFMCS.FindMCS([exp_mol, mol], maximizeBonds=False)
        # put bonds before atoms because lexsort works backwards
        match_results.append((mcs.numBonds, mcs.numAtoms))

    match_results = np.asarray(match_results)
    sort_idx = np.lexsort(-match_results.T)

    return(sort_idx)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-exp', required=True,
        help='JSON file giving experimental results.')
    parser.add_argument('-x', required=True,
        help='CSV file with crystal compound information.')
    parser.add_argument('-x_dir', required=True,
        help='Directory with crystal structures.')
    parser.add_argument('-loop', required=True, help='Spruce loop_db file.')
    parser.add_argument('-mcs', required=True, help='File with MCS results.')

    ## Output arguments
    parser.add_argument('-o', required=True, help='Main output directory.')
    parser.add_argument('-cache', help=('Cache directory (will use .cache in '
        'output directory if not specified).'))

    ## Performance arguments
    parser.add_argument('-n', default=1, type=int,
        help='Number of processors to use.')

    ## Filtering options
    parser.add_argument('-achiral', action='store_true',
        help='Whether to filter to only include achiral molecules.')
    parser.add_argument('-ep', action='store_true',
        help='Input data is in EnantiomerPairList format.')

    return(parser.parse_args())

def main():
    args = get_args()

    ## Load all compounds with experimental data and filter to only achiral
    ##  molecules (to start)
    if args.ep:
        exp_compounds = [c for ep in EnantiomerPairList(
            **json.load(open(args.exp, 'r'))).pairs \
            for c in (ep.active, ep.inactive)]
    else:
        exp_compounds = [c for c in ExperimentalCompoundDataUpdate(
            **json.load(open(args.exp, 'r'))).compounds if c.smiles is not None]
        if args.achiral:
            exp_compounds = np.asarray([c for c in exp_compounds if c.achiral])

    ## Find relevant crystal structures
    xtal_compounds = parse_xtal(args.x, args.x_dir)

    print(f'{len(exp_compounds)} experimental compounds')
    print(f'{len(xtal_compounds)} crystal structures')

    compound_ids, xtal_ids, res = pkl.load(open(args.mcs, 'rb'))
    ## Make sure all molecules line up
    assert len(exp_compounds) == len(compound_ids)
    assert len(xtal_compounds) == len(xtal_ids)
    assert all([exp_compounds[i].compound_id == compound_ids[i] \
        for i in range(len(compound_ids))])
    assert all([xtal_compounds[i].dataset == xtal_ids[i] \
        for i in range(len(xtal_ids))])

    docking_systems = build_docking_systems(exp_compounds, xtal_compounds, res)

    if args.cache is None:
        cache_dir = f'{args.o}/.cache/'
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
    else:
        cache_dir = args.cache
    print('Running docking', flush=True)
    n_procs = min(args.n, mp.cpu_count(), len(exp_compounds))
    featurizer = OEPositDockingFeaturizer(cache_dir=cache_dir,
        output_dir=args.o, loop_db=args.loop, n_processes=n_procs)
    featurizer.featurize(docking_systems)

if __name__ == '__main__':
    main()
