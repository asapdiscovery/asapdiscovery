"""
The purpose of this script is to make a multi-ligand SDF file which would be an input to the run_docking_oe.py script.
Currently, the point is to process the fragalysis dataset.
Example Usage:
    python combine_and_label_mpro_sdf.py \
    -csv /Users/alexpayne/Scientific_Projects/mers-drug-discovery/mpro-paper-ligand/extra_files/Mpro_compound_tracker_csv.csv \
    -s /Users/alexpayne/Scientific_Projects/mers-drug-discovery/mpro-paper-ligand/aligned/ \
    -o /Volumes/Rohirrim/local_test/sars_docking/fragalysis_correct_bond_orders.sdf
"""
import argparse, glob
import numpy as np
from asapdiscovery.data.utils import check_filelist_has_elements
from asapdiscovery.data.fragalysis import parse_fragalysis
from asapdiscovery.data.openeye import oechem, save_openeye_sdfs


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-csv",
        "--xtal_csv",
        required=True,
        help="Path to fragalysis metadata.csv.",
    )
    parser.add_argument(
        "-s",
        "--structure_dir",
        required=True,
        help="Path to fragalysis structure directory.",
    )
    parser.add_argument("--glob_str", help="Glob string to use to find the sdf files.")
    parser.add_argument(
        "-o",
        "--sdf_fn",
        required=True,
        help="Path to output multi-object sdf file that will be created",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Parsing '{args.xtal_csv}'")
    xtal_compounds = parse_fragalysis(
        args.xtal_csv,
        args.structure_dir,
    )
    print(f"Example: \n{xtal_compounds[0]}")

    xtal_compounds_array = np.array(xtal_compounds)
    dataset_array = np.array([cmpd.dataset for cmpd in xtal_compounds])

    # Load in SDF files
    if args.glob_str:
        sdfs = list([f for f in glob.glob(args.glob_str) if f.endswith(".sdf")])
        check_filelist_has_elements(sdfs)
        print(f"Loading {len(sdfs)} SDF files")
        from tqdm.notebook import tqdm
        from asapdiscovery.data.openeye import load_openeye_sdf

        mols = [load_openeye_sdf(sdf) for sdf in tqdm(sdfs)]

        import re

        # Make OEGraphMol for each compound and include some of the data
        print(f"Creating {len(mols)} OEGraphMol objects")
        for mol in tqdm(mols):
            cmplx = mol.GetTitle()
            xtal_pat = r"Mpro-.*?_[0-9][A-Z]"
            dataset = re.search(xtal_pat, cmplx)[0]
            cmpd = xtal_compounds_array[dataset_array == dataset][0]
            oechem.OESetSDData(mol, "SMILES", cmpd.smiles)
            oechem.OESetSDData(mol, "Dataset", cmpd.dataset)
            oechem.OESetSDData(mol, "Compound_ID", cmpd.compound_id)
            mol.SetTitle(cmpd.compound_id)
    else:
        raise NotImplementedError("Must provide glob_str")

    print(f"Saving to {args.sdf_fn}")
    save_openeye_sdfs(mols, args.sdf_fn)


if __name__ == "__main__":
    main()
