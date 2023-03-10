"""
Load in oedu design unit files, and write out the ligand sdf and protein-only pdb.
"""
import argparse
from pathlib import Path

import oechem
import oedocking
from asapdiscovery.data.openeye import (
    save_openeye_pdb,
    save_openeye_sdf,
    split_openeye_design_unit,
)
from tqdm import tqdm


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to directory of directories of input oedus.",
    )
    parser.add_argument(
        "-g", "--glob_str", default="*/*.oedu", help="String used to find files"
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Path to output_dir to save sdf files to. "
        "If not provided, default will be to put in the source directories.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    dir_path = Path(args.structure_dir)

    fns = list(dir_path.glob(args.glob_str))
    if not len(fns) > 0:
        raise Exception(
            f"No files found by {dir_path.resolve()}.glob({args.glob_str})!"
        )

    for fn in tqdm(fns):

        # Load in design units
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(fn), du)

        # Get name of complex from parent directory name
        complex_id = fn.parent.name

        # Use this function to split up design unit into different components
        lig, prot, complex = split_openeye_design_unit(du, lig_title=complex_id)

        # Get output names
        sdf_path = fn.parent / f"{complex_id}.sdf"
        prot_only = fn.parent / f"{complex_id}_protein.pdb"

        # Write out outputs
        save_openeye_sdf(lig, str(sdf_path))
        save_openeye_pdb(prot, str(prot_only))


if __name__ == "__main__":
    main()
