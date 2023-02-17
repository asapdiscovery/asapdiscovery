import argparse
import os

from asapdiscovery.data.schema import (
    CrystalCompoundData,
    PDBStructure,
)
from asapdiscovery.data.pdb import load_pdbs_from_yaml
from asapdiscovery.data.utils import (
    parse_experimental_compound_data,
    parse_fragalysis_data,
)


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-exp", required=True, help="CSV file with experimental data."
    )
    parser.add_argument(
        "-x", required=True, help="CSV file with crystal compound information."
    )
    parser.add_argument(
        "-x_dir", required=True, help="Directory with crystal structures."
    )
    parser.add_argument(
        "-o_dir", required=True, help="Directory to output files"
    )
    parser.add_argument(
        "-y", default="mers-structures.yaml", help="MERS structures yaml file"
    )
    parser.add_argument(
        "-m_dir", required=True, help="MERS structure directory"
    )
    return parser.parse_args()


def main():
    args = get_args()

    ligands = parse_experimental_compound_data(args.exp)
    cmpd_ids = [lig.compound_id for lig in ligands]
    sars_xtals = parse_fragalysis_data(args.x, args.x_dir, cmpd_ids, args.o_dir)
    pdb_list = load_pdbs_from_yaml(args.y)
    pdb_fn_dict = {
        pdb: os.path.join(
            args.m_dir, f"{pdb}_aligned_to_frag_ref_chainA_protein.pdb"
        )
        for pdb in pdb_list
    }
    mers_structures = [
        PDBStructure(pdb_id=pdb, str_fn=fn) for pdb, fn in pdb_fn_dict.items()
    ]

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


if __name__ == "__main__":
    main()
