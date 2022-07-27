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
from covid_moonshot_ml.docking.docking import parse_exp_cmp_data, parse_fragalysis_data


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

    ## Convert to schema
    exp_data_compounds = [
        ExperimentalCompoundData(
            compound_id=r["External ID"],
            smiles=r["SMILES"],
            experimental_data={
                "IC50": r["IC50"],
                "IC50_range": r["IC50_range"],
            },
        )
        for _, r in exp_df.iterrows()
    ]

    ## Dump JSON file
    with open(args.o, "w") as fp:
        fp.write(
            ExperimentalCompoundDataUpdate(compounds=exp_data_compounds).json()
        )


if __name__ == '__main__':
    main()
