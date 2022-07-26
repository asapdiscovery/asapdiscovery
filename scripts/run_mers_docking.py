import argparse
import os
import sys
import pandas as pd

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from covid_moonshot_ml.docking.docking import build_docking_systems,\
    parse_xtal, run_docking
from covid_moonshot_ml.datasets.utils import get_sdf_fns_from_dataset_list


################################################################################
# def get_args():
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-exp', required=True,
#                         help='CSV file with experimental data.')
#     parser.add_argument('-x', required=True,
#                         help='CSV file with crystal compound information.')
#     parser.add_argument('-x_dir', required=True,
#                         help='Directory with crystal structures.')
#     parser.add_argument('-d', help='Directory name to put the structures')
#     parser.add_argument('-y', default="mers-structures.yaml",
#                         help='MERS structures yaml file')
#     parser.add_argument('-r', default=None,
#                         help='Path to pdb reference file to align to')
#     parser.add_argument('-n', default=None, help='Name of reference')
#     return (parser.parse_args())


def main():
    # args = get_args()
    exp_data_fn = "COVID_Moonshot_Takeda_panCorona_enzyme_measurements.csv"
    sars2_cmpds_fn = "/Users/alexpayne/lilac-mount-point/fragalysis/extra_files/Mpro_compound_tracker_csv.csv"
    fragalysis_dir = "/Users/alexpayne/lilac-mount-point/fragalysis/aligned"

    exp_data = pd.read_csv(exp_data_fn).fillna("")
    sars2_structures = pd.read_csv(sars2_cmpds_fn)
    sars2_filtered = sars2_structures[sars2_structures['Compound ID'].isin(exp_data['External ID'])]

    mols_wo_sars2_xtal = sars2_filtered[sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES"]]
    mols_w_sars2_xtal = sars2_filtered[~sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES", "Dataset"]]

    mols_w_sars2_xtal.to_csv("mers_ligands_with_SARS2_structures.csv",
                             index=False)
    mols_wo_sars2_xtal.to_csv("mers_ligands_without_SARS2_structures.csv",
                              index=False)

    mols_w_sars2_xtal_datasets = mols_w_sars2_xtal["Dataset"].tolist()
    fns = get_sdf_fns_from_dataset_list(fragalysis_dir,
                                  mols_w_sars2_xtal_datasets)
    print(fns.values())


if __name__ == '__main__':
    main()
