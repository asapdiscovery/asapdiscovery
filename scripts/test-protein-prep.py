"""
Function to test implementation of ligand filtering
"""

import sys, os, argparse, yaml

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from covid_moonshot_ml.modeling import (
    align_receptor,
    prep_receptor,
    du_to_complex,
)
from covid_moonshot_ml.datasets.utils import (
    save_openeye_pdb,
    add_seqres,
    seqres_to_res_string,
)


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_prot",
        required=True,
        type=str,
        help="Path to pdb file of protein to prep.",
    )
    parser.add_argument(
        "-r",
        "--ref_prot",
        required=True,
        type=str,
        help="Path to reference pdb to align to.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Path to output_dir.",
    )
    parser.add_argument(
        "-l",
        "--loop_db",
        required=False,
        type=str,
        help="Path to loop database.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    base_file_name = os.path.splitext(os.path.split(args.input_prot)[1])[0]
    print(base_file_name)
    out_name = os.path.join(args.output_dir, base_file_name)

    ## first add standard seqres info

    with open("../data/SEQRES.yaml") as f:
        seqres_dict = yaml.safe_load(f)
    seqres = seqres_dict["MERS"]["SEQRES"]

    seq_str = seqres_to_res_string(seqres)

    seqres_pdb = f"{out_name}_seqres.pdb"
    add_seqres(args.input_prot, seqres_str="", pdb_out=seqres_pdb)

    for mobile_chain in ["A", "B"]:
        chain_name = f"{out_name}_chain{mobile_chain}"
        initial_prot = align_receptor(
            input_prot=seqres_pdb,
            ref_prot=args.ref_prot,
            dimer=True,
            mobile_chain=mobile_chain,
            ref_chain="A",
        )

        aligned_fn = f"{chain_name}_aligned.pdb"
        save_openeye_pdb(initial_prot, aligned_fn)

        # site_residue = "HIS:41: :A"
        # design_units = prep_receptor(
        #     initial_prot,
        #     site_residue=site_residue,
        #     sequence=seq_str,
        #     loop_db=args.loop_db,
        # )
        # for i, du in enumerate(design_units):
        #     print(i, du)
        #     complex_mol = du_to_complex(du)
        #     prepped_fn = f"{chain_name}_prepped.pdb"
        #     save_openeye_pdb(complex_mol, prepped_fn)

    # from kinoml.features.protein import OEProteinStructureFeaturizer
    # from kinoml.core.proteins import Protein, KLIFSKinase
    # from kinoml.core.systems import ProteinSystem, ProteinLigandComplex
    #
    # systems = []
    # protein = Protein.from_file(file_path=args.input_prot, name="7DR8")
    # protein.sequence = seq_str
    # system = ProteinSystem(components=[protein])
    # systems.append(system)
    # featurizer = OEProteinStructureFeaturizer(
    #     loop_db=args.loop_db,
    #     output_dir=args.output_dir,
    #     use_multiprocessing=False,
    # )
    # featurizer.featurize(systems)


if __name__ == "__main__":
    main()
