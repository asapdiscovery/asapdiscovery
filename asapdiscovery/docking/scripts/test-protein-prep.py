"""
Function to test implementation of protein-prep functions
"""

import sys, os, argparse, yaml

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from asapdiscovery.docking.modeling import (
    align_receptor,
    prep_receptor,
    du_to_complex,
    mutate_residues,
)
from asapdiscovery.data.utils import (
    edit_pdb_file,
    seqres_to_res_list,
)
from asapdiscovery.data.openeye import save_openeye_pdb, load_openeye_pdb
from openeye import oechem


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
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        default=None,
        type=str,
        help="Path to yaml file of SEQRES.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    base_file_name = os.path.splitext(os.path.split(args.input_prot)[1])[0]
    print(base_file_name)
    out_name = os.path.join(args.output_dir, base_file_name)

    if args.seqres_yaml:
        ## first add standard seqres info

        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]

        ## Get a list of 3-letter codes for the sequence
        res_list = seqres_to_res_list(seqres)

        ## Generate a new pdb file with the SEQRES we want
        seqres_pdb = f"{out_name}_01seqres.pdb"
        edit_pdb_file(args.input_prot, seqres_str=seqres, pdb_out=seqres_pdb)

        ## Load in the pdb file as an OE object
        seqres_prot = load_openeye_pdb(seqres_pdb)

        ## Mutate the residues to match the residue list
        initial_prot = mutate_residues(seqres_prot, res_list)
        mutated_fn = f"{out_name}_02mutated.pdb"
        save_openeye_pdb(initial_prot, mutated_fn)
    else:
        initial_prot = load_openeye_pdb(args.input_prot)

    ## For each chain, align the receptor to the reference while keeping both chains.
    for mobile_chain in ["A", "B"]:
        print(f"Running on chain {mobile_chain}")
        aligned_prot = align_receptor(
            input_prot=initial_prot,
            ref_prot=args.ref_prot,
            dimer=True,
            mobile_chain=mobile_chain,
            ref_chain="A",
        )
        aligned_fn = f"{out_name}_03aligned_chain{mobile_chain}.pdb"
        save_openeye_pdb(aligned_prot, aligned_fn)

        ## Prep the receptor using various SPRUCE options
        site_residue = "HIS:41: :A"
        design_units = prep_receptor(
            aligned_prot,
            site_residue=site_residue,
            loop_db=args.loop_db,
        )

        ## Because of the object I'm using, it returns the design units as a list
        ## There should only be one but just in case I'm going to write out all of them
        for i, du in enumerate(design_units):
            print(i, du)

            ## First save the design unit itself
            oechem.OEWriteDesignUnit(
                f"{out_name}_04prepped_chain{mobile_chain}.oedu", du
            )

            ## Then save as a PDB file
            complex_mol = du_to_complex(du)
            prepped_fn = f"{out_name}_04prepped_chain{mobile_chain}.pdb"
            save_openeye_pdb(complex_mol, prepped_fn)


if __name__ == "__main__":
    main()
