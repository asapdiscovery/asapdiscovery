import argparse
import os
import sys
import pandas as pd

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from covid_moonshot_ml.docking.docking import build_docking_systems, \
    parse_xtal, run_docking
from covid_moonshot_ml.schema import CrystalCompoundData, \
    ExperimentalCompoundData, PDBStructure
from covid_moonshot_ml.datasets.utils import get_sdf_fn_from_dataset
from covid_moonshot_ml.datasets.pdb import load_pdbs_from_yaml


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
    parser.add_argument('-y', default="mers-structures.yaml",
                        help='MERS structures yaml file')
    parser.add_argument('-m_dir', required=True,
                        help="MERS structure directory")
    return (parser.parse_args())


def main():
    args = get_args()
    #
    # ## Load in compound data
    # exp_data = pd.read_csv(args.exp).fillna("")
    # sars2_structures = pd.read_csv(args.x)
    #
    # ## Filter fragalysis dataset by the compounds we want to test
    # sars2_filtered = sars2_structures[sars2_structures['Compound ID'].isin(exp_data['External ID'])]
    #
    # ## Split dataset bassed on whehter or not there is an ID
    # ## Currently I'm not saving the IC50 information but we could do that
    # mols_wo_sars2_xtal = sars2_filtered[sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES", "Dataset"]]
    # mols_w_sars2_xtal = sars2_filtered[~sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES", "Dataset"]]
    #
    # ## Use utils function to get sdf file from dataset
    # mols_w_sars2_xtal["SDF"] = mols_w_sars2_xtal["Dataset"].apply(get_sdf_fn_from_dataset,
    #                                                               fragalysis_dir=args.x_dir)
    #
    # ## Save csv files for each dataset
    # mols_wo_sars2_xtal.to_csv(os.path.join(args.o_dir, "mers_ligands_without_SARS2_structures.csv"),
    #                           index=False)
    #
    # mols_w_sars2_xtal.to_csv(os.path.join(args.o_dir, "mers_ligands_with_SARS2_structures.csv"),
    #                          index=False)
    #
    # # mols_w_sars2_xtal.index = mols_w_sars2_xtal["Compound ID"]
    # ligand_args = mols_w_sars2_xtal.to_dict('index')
    # print(ligand_args)
    #
    # exp_cmpds = exp_data.to_dict('index')
    # sars_xtal_dict = mols_w_sars2_xtal.to_dict('index')
    #
    # ## Construct sars_xtal list
    # sars_xtals = {}
    # for data in sars_xtal_dict.values():
    #     dataset = data["Dataset"]
    #     sars_xtals[dataset] = CrystalCompoundData(
    #         smiles=data["SMILES"],
    #         compound_id=data["Compound ID"],
    #         dataset=dataset,
    #         sdf_fn=get_sdf_fn_from_dataset(dataset, args.x_dir)
    #     )
    # print(sars_xtals)
    #
    # ligands_w_sars_structs = []
    # ligands_wo_sars_structs = []
    # lig_dataset_map = {}
    # for data in exp_cmpds.values():
    #     cmpd_id = data["External ID"]
    #     smiles = data["SMILES"]
    #     ligand = ExperimentalCompoundData(compound_id=cmpd_id, smiles=smiles)
    #     datasets = [data["Dataset"] for data in sars_xtal_dict.values() if data["Compound ID"] == cmpd_id]
    #     if len(datasets) == 0:
    #         ligands_wo_sars_structs.append(ligand)
    #     elif len(datasets) == 1:
    #         ligands_w_sars_structs.append(ligand)
    #         lig_dataset_map[cmpd_id] = sars_xtals[datasets[0]]
    #     elif len(datasets) >= 1:
    #         for dataset in datasets:
    #             if '-P' in dataset:
    #                 # print(dataset)
    #                 ligands_w_sars_structs.append(ligand)
    #                 lig_dataset_map[cmpd_id] = sars_xtals[dataset]
    #
    # print(lig_dataset_map)
    # print(len(ligands_w_sars_structs), ligands_w_sars_structs)
    # print(len(ligands_wo_sars_structs), ligands_wo_sars_structs)
    # # for lig in ligands_w_sars_structs:
    #
    # # ligands = []
    # # for ligand in mols_w_sars2_xtal["Compound ID"].to_list():
    # #     print(ligand)
    # mers_structures = []
    # pdb_list = load_pdbs_from_yaml(args.y)
    # for pdb in pdb_list:
    #     mers_fn = os.path.join(args.m_dir, f"{pdb}_aligned_to_frag_ref_chainA_protein.pdb")
    #     assert os.path.exists(mers_fn)
    #
    #     mers_structures.append(
    #         PDBStructure(
    #             pdb_id=pdb,
    #             str_fn=mers_fn
    #         )
    #     )
    # print(mers_structures)
    #
    # ## could use itertools but not really necessary yet?
    # combinations = [(lig, pdb) for lig in ligands_w_sars_structs for pdb in mers_structures]
    # print(f"Running {len(combinations)} docking combinations")
    # for lig, pdb in combinations:
    #     #
    #     sars_xtal = lig_dataset_map[lig.compound_id]
    #     print(pdb.pdb_id, lig.compound_id, sars_xtal.dataset)
    #     # print(sars_xtal.dataset, sars_xtal.sdf_fn)

    from covid_moonshot_ml.docking.docking import parse_exp_cmp_data
    parse_exp_cmp_data(args.exp,
                       args.x,
                       args.x_dir)


    ## TODO
    # Run MCSS on the mols without SARS2 ligands and return a dataset for each Compound ID
    # Get dictionary mapping a sars_dataset to each exp_ligand (can read from csv file)
    # Construct three objects:
    # exp_ligand(SMILES, Compound ID, IC50s)
    # mers_structure(PDB ID, filepath)
    # sars_dataset(dataset_name, SDF file, Compound ID, SMILES)
    # For each exp_ligand, for each mers_structure:
    #


if __name__ == '__main__':
    main()
