import sys, os, argparse
import pickle as pkl
import numpy as np
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../')
from covid_moonshot_ml.docking.docking import build_docking_systems, \
    parse_xtal, run_docking
from covid_moonshot_ml.datasets.utils import load_openeye_pdb, \
    get_ligand_rmsd
from covid_moonshot_ml.schema import ExperimentalCompoundDataUpdate, \
    EnantiomerPairList, DockingDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-d', required=True,
        help='Directory containing docking results.')

    return parser.parse_args()

def main():
    args = get_args()

    pkl_fn = os.path.join(args.d, 'mcs_sort_index.pkl')

    dd = DockingDataset(pkl_fn=pkl_fn, dir_path=args.d)
    dd.read_pkl()

    fn = os.path.join(args.d, 'ADA-UCB-6c2cb422-1/kinoml_OEDockingFeaturizer_MPRO_Mpro-P0764_0B_seqres_chainB_ADA-UCB-6c2cb422-1_complex_1.pdb')
    ref = os.path.join('/Users/alexpayne/lilac-mount-point/fragalysis/aligned/Mpro-P2005_0A/Mpro-P2005_0A_bound.pdb')
    mobile = load_openeye_pdb(fn)
    reference = load_openeye_pdb(ref)
    print(get_ligand_rmsd(mobile, reference))

    # dd.calculate_RMSDs() ## doesn't exist yet
    # dd.write_csv()




    # compound_ids, xtal_ids, res = pkl.load(open(pkl_fn, 'rb'))

    # print(res)


if __name__ == '__main__':
    main()