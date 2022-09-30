"""
Function to test implementation of ligand filtering
"""

import sys, os, argparse

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from covid_moonshot_ml.modeling import prep_receptor, du_to_complex
from covid_moonshot_ml.datasets.utils import save_openeye_pdb


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    # parser.add_argument(
    #     "-f",
    #     "--fragalysis_dir",
    #     required=True,
    #     type=str,
    #     help="Path to fragalysis directory.",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--csv_file",
    #     required=True,
    #     type=str,
    #     help="Path to csv file containing compound info.",
    # )
    # parser.add_argument(
    #     "-s",
    #     "--smarts_queries",
    #     default="../data/smarts_queries.csv",
    #     type=str,
    #     help="Path to csv file containing smarts queries.",
    # )

    return parser.parse_args()


def main():
    args = get_args()
    input_prot = "/Volumes/Rohirrim/local_test/mers_hallucination_hybrid/mers_structures/rcsb_7DR8.pdb"
    ref_prot = "/Users/alexpayne/Scientific_Projects/mers-drug-discovery/Mpro-paper-ligand/extra_files/reference.pdb"
    for mobile_chain in ["A", "B"]:
        out_fn = f"/Volumes/Rohirrim/local_test/mers_hallucination_hybrid/mers_structures/rcsb_7DR8_aligned_dimer{mobile_chain}"
        initial_prot = prep_receptor(
            input_prot=input_prot,
            ref_prot=ref_prot,
            dimer=True,
            mobile_chain=mobile_chain,
            ref_chain="A",
        )
        ## Get protein+lig complex in molecule form and save
        # complex_mol = du_to_complex(initial_prot)
        save_openeye_pdb(initial_prot, f"{out_fn}.pdb")


if __name__ == "__main__":
    main()
