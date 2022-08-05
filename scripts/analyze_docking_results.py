import sys, os, argparse
import pickle as pkl
import numpy as np
import pandas as pd

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../')
from covid_moonshot_ml.docking.docking import build_docking_systems, \
    parse_xtal, run_docking
from covid_moonshot_ml.datasets.utils import load_openeye_pdb, \
    get_ligand_rmsd_openeye, get_ligand_RMSD_mdtraj, load_openeye_sdf
from covid_moonshot_ml.schema import ExperimentalCompoundDataUpdate, \
    EnantiomerPairList
from covid_moonshot_ml.docking.analysis import DockingDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-d', required=True,
        help='Directory containing docking results.')

    parser.add_argument('-f', required=True,
                        help='Directory containing fragalysis data.')

    return parser.parse_args()

def main():
    args = get_args()

    pkl_fn = os.path.join(args.d, 'mcs_sort_index.pkl')

    dd = DockingDataset(pkl_fn=pkl_fn, dir_path=args.d)
    dd.read_pkl()
    dd.analyze_docking_results(args.f,
                               "docking_results.csv",
                               test=False)
if __name__ == '__main__':
    main()