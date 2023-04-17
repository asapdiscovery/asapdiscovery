"""
The goal of this function is to take an SDF file of molecules, align them in 2D, and display them
Example Usage:
python display_aligned_2d_molecules.py
    -s ~/asap-datasets/amines_to_dock_small.sdf
    -o ~/asap-datasets/test_ligand_visualization

"""
import argparse
import os
import sys

from openeye import oechem, oedepict

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from asapdiscovery.data.openeye import load_openeye_sdf, load_openeye_sdfs
from asapdiscovery.dataviz.molecules import display_openeye_ligand


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-s",
        "--sdf_fn",
        required=True,
        help="SDF file containing molecules of interest.",
    )
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "-r",
        "--ref_mol",
        help="Reference molecule to be aligned to. If blank, first molecule in sdf file is used",
    )
    parser.add_argument(
        "--do_not_align",
        action="store_true",
        default=False,
        help="If flag is passed, script will skip the "
        "mcss-based alignment and just write out the positions in the sdf file.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    mols = load_openeye_sdfs(args.sdf_fn)
    if args.ref_mol:
        ref = load_openeye_sdf(args.ref_mol)
    if not args.ref_mol:
        ref = mols[0]

    for mol in mols:
        # Prepare mcss
        if not args.do_not_align:
            out_fn = (
                f"{os.path.join(args.output_dir, mol.GetTitle())}_aligned.png".replace(
                    " ", "_"
                )
            )
            # this code is mostly taken from the mcsalign2d.py example from openeye
            # <https://docs.eyesopen.com/toolkits/python/depicttk/examples_summary_mcsalign2D.html>
            mcss = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)
            atomexpr = oechem.OEExprOpts_DefaultAtoms
            bondexpr = oechem.OEExprOpts_DefaultBonds
            mcss.Init(ref, atomexpr, bondexpr)
            mcss.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())

            # Prepare openeye aligned depiction
            alignres = oedepict.OEPrepareAlignedDepiction(mol, mcss)

            # Write out image
            if alignres.IsValid():
                display_openeye_ligand(mol, out_fn=out_fn, aligned=True)
        else:
            out_fn = f"{os.path.join(args.output_dir, mol.GetTitle())}.png".replace(
                " ", "_"
            )
            display_openeye_ligand(mol, out_fn=out_fn)


if __name__ == "__main__":
    main()
