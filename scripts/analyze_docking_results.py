import sys, os, argparse
import pickle as pkl
import numpy as np
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../')
from covid_moonshot_ml.docking.docking import build_docking_systems, \
    parse_xtal, run_docking
from covid_moonshot_ml.datasets.utils import load_openeye_pdb, \
    get_ligand_rmsd, get_ligand_RMSD_mdtraj
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

    # mobile_fn = os.path.join(args.d, 'ADA-UCB-6c2cb422-1/kinoml_OEDockingFeaturizer_MPRO_Mpro-P0764_0B_seqres_chainB_ADA-UCB-6c2cb422-1_complex_1.pdb')
    # mobile_fn = os.path.join(args.d, 'ADA-UCB-6c2cb422-1/kinoml_OEDockingFeaturizer_MPRO_Mpro-P0394_0A_seqres_chainA_ADA-UCB-6c2cb422-1_complex_2.pdb')
    # mobile_fn = os.path.join(args.d, 'ADA-UCB-6c2cb422-1/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2210_0B_seqres_chainB_ADA-UCB-6c2cb422-1_complex_6.pdb')
    # ref_fn = os.path.join('/Users/alexpayne/lilac-mount-point/fragalysis/aligned/Mpro-P2005_0A/Mpro-P2005_0A_bound.pdb')

    ref_fn = os.path.join('/Users/alexpayne/lilac-mount-point/fragalysis/aligned/Mpro-P2291_0B/Mpro-P2291_0B_bound.pdb')
    mobile_fn = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2291_0A_seqres_chainA_EDJ-MED-43f8f7d6-4_complex.pdb')
    mobile_fn2 = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2468_0B_seqres_chainB_EDJ-MED-43f8f7d6-4_complex_2.pdb')
    mobile_fn3 = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2074_0B_seqres_chainB_EDJ-MED-43f8f7d6-4_complex_7.pdb')
    mobile_fn4 = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2039_0B_seqres_chainB_EDJ-MED-43f8f7d6-4_complex_6.pdb')


    mobile = load_openeye_pdb(mobile_fn)
    reference = load_openeye_pdb(ref_fn)
    # print(get_ligand_rmsd(mobile, reference))

    for mobile_fn in [ref_fn, mobile_fn, mobile_fn2, mobile_fn3, mobile_fn4]:
        print(mobile_fn)
        get_ligand_RMSD_mdtraj(ref_fn, mobile_fn)

    # dd.calculate_RMSDs() ## doesn't exist yet
    # dd.write_csv()

    # compound_ids, xtal_ids, res = pkl.load(open(pkl_fn, 'rb'))

    # print(res)


if __name__ == '__main__':
    main()