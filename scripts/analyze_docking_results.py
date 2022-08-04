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
    EnantiomerPairList, DockingDataset


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
    dd.analyze_docking_results(args.d, "docking_results.csv")
    #
    # rmsds = []
    # ref_fns = []
    # for idx in range(len(cmp_ids)):
    #     cmp_id = cmp_ids[idx]
    #     xtal_id = xtal_ids[idx]
    #     chain = chain_ids[idx]
    #     mcss_rank = mcss_ranks[idx]
    #
    #     cmp_dir = os.path.join(args.d, cmp_id)
    #
    #     sdf_fn = os.path.join(cmp_dir, sdf_fns[idx])
    #     ref_fn = os.path.join(args.f, ref_fn_dict[cmp_id])
    #
    #     print(f"Running rmsd calc on {sdf_fn} compared to {ref_fn}")
    #     ref = load_openeye_sdf(ref_fn)
    #     mobile = load_openeye_sdf(sdf_fn)
    #
    #     rmsd = get_ligand_rmsd_openeye(ref, mobile)
    #
    #     ref_fns.append(ref_fn)
    #     rmsds.append(rmsd) ## convert to angstroms
    #
    #
    # df = pd.DataFrame(
    #     {"Compound_ID": cmp_ids,
    #      "Crystal ID": xtal_ids,
    #      "Chain ID": chain_ids,
    #      "MCSS Rank": mcss_ranks,
    #      "SDF Filename": sdf_fns,
    #      "Reference SDF": ref_fns,
    #      "RMSD": rmsds
    #
    #      }
    # )
    #
    # # print(df.head)
    # df.to_csv("docking_results.csv")




        # print(ref_sdf_fn)



        # for xtal_id in dd.xtal_ids:
        #     print(xtal_id)
            # docked_system = f"kinoml_OEDockingFeaturizer_MPRO_{xtal_id}_seqres"

    # mobile_fn = os.path.join(args.d, 'ADA-UCB-6c2cb422-1/kinoml_OEDockingFeaturizer_MPRO_Mpro-P0764_0B_seqres_chainB_ADA-UCB-6c2cb422-1_complex_1.pdb')
    # mobile_fn = os.path.join(args.d, 'ADA-UCB-6c2cb422-1/kinoml_OEDockingFeaturizer_MPRO_Mpro-P0394_0A_seqres_chainA_ADA-UCB-6c2cb422-1_complex_2.pdb')
    # mobile_fn = os.path.join(args.d, 'ADA-UCB-6c2cb422-1/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2210_0B_seqres_chainB_ADA-UCB-6c2cb422-1_complex_6.pdb')
    # ref_fn = os.path.join('/Users/alexpayne/lilac-mount-point/fragalysis/aligned/Mpro-P2005_0A/Mpro-P2005_0A_bound.pdb')

    # ref_fn = os.path.join('/Users/alexpayne/lilac-mount-point/fragalysis/aligned/Mpro-P2291_0B/Mpro-P2291_0B_bound.pdb')
    # mobile_fn = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2291_0A_seqres_chainA_EDJ-MED-43f8f7d6-4_complex.pdb')
    # mobile_fn2 = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2468_0B_seqres_chainB_EDJ-MED-43f8f7d6-4_complex_2.pdb')
    # mobile_fn3 = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2074_0B_seqres_chainB_EDJ-MED-43f8f7d6-4_complex_7.pdb')
    # mobile_fn4 = os.path.join(args.d, 'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2039_0B_seqres_chainB_EDJ-MED-43f8f7d6-4_complex_6.pdb')

    # ref_fn = "/Users/alexpayne/lilac-mount-point/fragalysis/aligned/Mpro-P2291_0B/Mpro-P2291_0B.sdf"
    # mobile_fn = os.path.join(args.d,
    #                          'EDJ-MED-43f8f7d6-4/kinoml_OEDockingFeaturizer_MPRO_Mpro-P2291_0A_seqres_chainA_EDJ-MED-43f8f7d6-4_ligand.sdf')
    #
    # ref = load_openeye_sdf(ref_fn)
    # mobile = load_openeye_sdf(mobile_fn)
    # print(get_ligand_rmsd_openeye(ref, mobile))



    # mobile = load_openeye_pdb(mobile_fn)
    # reference = load_openeye_pdb(ref_fn)
    # print(get_ligand_rmsd(mobile, reference))

    # for mobile_fn in [mobile_fn, mobile_fn2, mobile_fn3, mobile_fn4]:
    #     print(mobile_fn)
    #     get_ligand_RMSD_mdtraj(ref_fn, mobile_fn)

    # dd.calculate_RMSDs() ## doesn't exist yet
    # dd.write_csv()

    # compound_ids, xtal_ids, res = pkl.load(open(pkl_fn, 'rb'))

    # print(res)


if __name__ == '__main__':
    main()