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
    #
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
    #         )
    #     )
    # print(mers_structures)
    #

    from covid_moonshot_ml.docking.docking import parse_exp_cmp_data, parse_fragalysis_data
    ligands = parse_exp_cmp_data(args.exp)
    cmpd_ids = [lig.compound_id for lig in ligands]
    sars_xtals = parse_fragalysis_data(args.x,
                                       args.x_dir,
                                       cmpd_ids,
                                       args.o_dir)
    pdb_list = load_pdbs_from_yaml(args.y)
    pdb_fn_dict = {pdb: os.path.join(args.m_dir, f"{pdb}_aligned_to_frag_ref_chainA_protein.pdb") for pdb in pdb_list}
    mers_structures = [PDBStructure(pdb_id=pdb, str_fn=fn) for pdb, fn in pdb_fn_dict.items()]

    ## could use itertools but not really necessary yet?
    combinations = [(lig, pdb) for lig in ligands for pdb in mers_structures]
    print(f"Running {len(combinations)} docking combinations")
    for lig, pdb in combinations[0]:
        sars_xtal = sars_xtals.get(lig.compound_id, CrystalCompoundData())
        if sars_xtal.sdf_fn:
            print(pdb.pdb_id, lig.compound_id, sars_xtal.sdf_fn)
        else:
            print(f"Skipping {pdb.pdb_id}, {lig.compound_id}")

        ## Profit?

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
