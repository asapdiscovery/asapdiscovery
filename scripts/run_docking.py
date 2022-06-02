import argparse
import json
import os
import pickle as pkl

from covid_moonshot_ml.docking.docking import build_docking_systems, \
    parse_xtal, run_docking
from covid_moonshot_ml.schema import ExperimentalCompoundDataUpdate, \
    EnantiomerPairList

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
            exp_compounds = [c for c in exp_compounds if c.achiral]

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
<<<<<<< HEAD:scripts/run_docking.py
    n_procs = min(args.n, len(exp_compounds))
    run_docking(cache_dir, args.o, args.loop, n_procs, docking_systems)
=======
    n_procs = min(args.n, mp.cpu_count(), len(exp_compounds))
    featurizer = OEDockingFeaturizer(cache_dir=cache_dir,
        output_dir=args.o, loop_db=args.loop, n_processes=n_procs)
    docking_systems = featurizer.featurize(docking_systems)

    ## Save docking and posit scores
    print('Docking scores:', flush=True)
    scores_dict = {
        'compound_id': [],
        'docking_score': [],
        'posit_score': []
    }
    for (c, s) in zip(exp_compounds, docking_systems):
        scores_dict['compound_id'].append(c.compound_id)
        try:
            scores_dict['docking_score'].append(
                s.featurizations['last']._topology.docking_score)
            scores_dict['posit_score'].append(
                s.featurizations['last']._topology.posit_probability)
        except AttributeError as e:
            print(dir(s.featurizations['last']._topology), flush=True)
            raise e

        print(c.compound_id, scores_dict['docking_score'][-1],
            scores_dict['posit_score'][-1], flush=True)

    scores_dict = pandas.DataFrame(scores_dict)
    scores_dict.to_csv(f'{args.o}/docking_scores.csv', index=False)
>>>>>>> master:covid_moonshot_ml/run_docking.py

if __name__ == '__main__':
    main()
