"""
The purpose of this script is to prepare a csv file containing a set of structures to prepare.
The idea here is to split up the process of preparing structures into two steps:
1. Prepare a csv file containing the structures to prepare and information related to the preparation
2. Actually prepare the structures using the csv file from step 1
"""
from pathlib import Path
from argparse import ArgumentParser
from asapdiscovery.data.fragalysis import parse_fragalysis
from asapdiscovery.data.schema import CrystalCompoundDataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-csv", "--fragalysis_csv", type=Path, required=True)
    parser.add_argument("-d", "--structure_dir", type=Path, required=True)
    parser.add_argument("--include_non_Pseries", action="store_true", required=False)
    parser.add_argument("-o", "--output_csv", type=Path, required=True)
    parser.add_argument("--protein_only", action="store_true", required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fragalysis_csv:
        p_only = False if args.include_non_Pseries else True
        if p_only:
            xtal_compounds = parse_fragalysis(
                args.fragalysis_csv,
                args.structure_dir,
                name_filter="Mpro-P",
                drop_duplicate_datasets=True,
            )
        else:
            xtal_compounds = parse_fragalysis(
                args.fragalysis_csv,
                args.structure_dir,
            )

        for xtal in xtal_compounds:
            # Get chain
            # The parentheses in this string are the capture group

            xtal.output_name = f"{xtal.dataset}_{xtal.compound_id}"

            frag_chain = xtal.dataset[-2:]

            # We also want the chain in the form of a single letter ('A', 'B'), etc
            xtal.active_site_chain = frag_chain[-1]

            # If we aren't keeping the ligands, then we want to give it a site residue to use
            if args.protein_only:
                xtal.active_site = f"His:41: :{xtal.active_site_chain}"

        xtal_dataset = CrystalCompoundDataset(compounds=xtal_compounds)
        xtal_dataset.to_csv(args.output_csv)


if __name__ == "__main__":
    main()
