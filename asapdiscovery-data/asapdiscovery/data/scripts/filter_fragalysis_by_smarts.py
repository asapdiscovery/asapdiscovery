"""
Filter the full fragalysis dataset to only include structures whose bound ligand
matches the given SMARTS string(s). Output is a CSV file with all kept
structures.
"""
import argparse
import pandas

from asapdiscovery.data.utils import (
    filter_docking_inputs,
    parse_fragalysis_data,
)

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-frag",
        "--frag_input_dir",
        required=True,
        help="Input fragalysis directory.",
    )
    parser.add_argument(
        "-o", "--out_file", required=True, help="Output CSV file."
    )
    parser.add_argument(
        "-s",
        "--smarts_filter",
        required=True,
        help="CSV file with SMARTS strings.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## First, parse the fragalysis directory into a dictionary of
    ##  CrystalCompoundData
    xtal_fn = f"{args.frag_input_dir}/extra_files/Mpro_compound_tracker_csv.csv"
    sars_xtals = parse_fragalysis_data(xtal_fn, args.frag_input_dir)

    ## For the compounds for which we have smiles strings, get a
    ##  dictionary mapping the Compound_ID to the smiles
    cmp_to_smiles_dict = {
        compound_id: data.smiles
        for compound_id, data in sars_xtals.items()
        if data.smiles
    }

    ## Filter based on the smiles using this OpenEye function
    filtered_inputs = filter_docking_inputs(
        smarts_queries=args.smarts_filter,
        docking_inputs=cmp_to_smiles_dict,
    )

    ## Build output df
    datasets = []
    compound_ids = []
    smiles = []
    for compound_id in filtered_inputs:
        xtal = sars_xtals[compound_id]
        datasets.append(xtal.dataset)
        compound_ids.append(compound_id)
        smiles.append(xtal.smiles)
    df = pandas.DataFrame(
        {"Dataset": datasets, "Compound ID": compound_ids, "SMILES": smiles}
    )
    df.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    main()
