import argparse
import os
import sys
import pandas as pd

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from covid_moonshot_ml.docking.docking import build_docking_systems, \
    parse_xtal, run_docking
from covid_moonshot_ml.datasets.utils import get_sdf_fn_from_dataset_list


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-exp', required=True,
                        help='CSV file with experimental data.')
    parser.add_argument('-x', required=True,
                        help='CSV file with crystal compound information.')
    parser.add_argument('-x_dir', required=True,
                        help='Directory with crystal structures.')
    parser.add_argument('-o_dir', required=True,
                        help="Directory to output files")
    return (parser.parse_args())


def main():
    args = get_args()

    ## Load in compound data
    exp_data = pd.read_csv(args.exp).fillna("")
    sars2_structures = pd.read_csv(args.x)

    ## Filter fragalysis dataset by the compounds we want to test
    sars2_filtered = sars2_structures[sars2_structures['Compound ID'].isin(exp_data['External ID'])]

    ## Split dataset bassed on whehter or not there is an ID
    ## Currently I'm not saving the IC50 information but we could do that
    mols_wo_sars2_xtal = sars2_filtered[sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES", "Dataset"]]
    mols_w_sars2_xtal = sars2_filtered[~sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES", "Dataset"]]

    ## Use utils function to get sdf file from dataset
    mols_w_sars2_xtal["SDF"] = mols_w_sars2_xtal["Dataset"].apply(get_sdf_fn_from_dataset_list,
                                                                  fragalysis_dir=args.x_dir)

    ## Save csv files for each dataset
    mols_wo_sars2_xtal.to_csv(os.path.join(args.o_dir, "mers_ligands_without_SARS2_structures.csv"),
                              index=False)

    mols_w_sars2_xtal.to_csv(os.path.join(args.o_dir, "mers_ligands_with_SARS2_structures.csv"),
                             index=False)


if __name__ == '__main__':
    main()
