"""
Function to test implementation of protein-prep functions
Example Usage:
python test-mpro-fragalysis-seqres-addition.py
    -i /Users/alexpayne/lilac-mount-point/asap-datasets/mpro_fragalysis_2022_10_12/aligned/Mpro-P3054_0A/Mpro-P3054_0A_bound.pdb
    -o /Users/alexpayne/lilac-mount-point/asap-datasets/test_protein_prep/
    -s /Users/alexpayne/Scientific_Projects/covid-moonshot-ml/metadata/mpro_sars2_seqres.yaml
"""

import argparse
import os
import sys
from tempfile import NamedTemporaryFile

import yaml

sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
from asapdiscovery.data.openeye import load_openeye_pdb, save_openeye_pdb
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.utils import edit_pdb_file, seqres_to_res_list
from asapdiscovery.docking.modeling import (
    align_receptor,
    du_to_complex,
    mutate_residues,
    prep_receptor,
)
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
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Path to output_dir.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        default=None,
        type=str,
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "-chain",
        "--active_site_chain",
        default="A",
        type=str,
        help="Chain letter to be used as the active site",
    )

    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    active_site_chain = args.active_site_chain

    ## Load PDB File

    xtal = CrystalCompoundData(
        str_fn=args.input_prot,
        active_site_chain=active_site_chain,
        # chains=values.get("chains", ["A", "B"]),
        # oligomeric_state=values.get("oligomeric_state", "dimer"),
        # protein_chains=values.get("protein_chains", ["A", "B"]),
    )

    if args.seqres_yaml:
        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]
    else:
        seqres = None

    ## Option to add SEQRES header
    if seqres:
        print("Editing PDB file")
        ## Get a list of 3-letter codes for the sequence
        res_list = seqres_to_res_list(seqres)

        ## Generate a new (temporary) pdb file with the SEQRES we want
        with NamedTemporaryFile(mode="w", suffix=".pdb") as tmp_pdb:
            ## Add the SEQRES
            edit_pdb_file(
                xtal.str_fn,
                seqres_str=seqres,
                edit_remark350=True,
                oligomeric_state=xtal.oligomeric_state,
                chains=xtal.chains,
                pdb_out=tmp_pdb.name,
            )

            ## Load in the pdb file as an OE object
            seqres_prot = load_openeye_pdb(tmp_pdb.name)

            save_openeye_pdb(
                seqres_prot, os.path.join(args.output_dir, "seqres_test.pdb")
            )

            initial_prot = seqres_prot
        mutate = True
    else:
        initial_prot = load_openeye_pdb(xtal.str_fn)
        mutate = False

    if mutate:
        print("Mutating to provided seqres")
        ## Mutate the residues to match the residue list
        initial_prot = mutate_residues(initial_prot, res_list, xtal.protein_chains)
        save_openeye_pdb(
            initial_prot, os.path.join(args.output_dir, "mutated_test.pdb")
        )


if __name__ == "__main__":
    main()
