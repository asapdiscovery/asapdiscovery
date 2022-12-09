"""
The goal of this function is to take an SDF file of molecules, align them in 2D, and display them
Example Usage:
python display_aligned_2d_molecules.py
    -s ~/asap-datasets/amines_to_dock_small.sdf
    -o ~/asap-datasets/test_ligand_visualization

"""
import os, sys, argparse

from openeye import oechem, oespruce, oedepict

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from asapdiscovery.data.openeye import load_openeye_sdf, load_openeye_sdfs

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-s",
        "--sdf_fn",
        required=True,
        help="SDF file containing molecules of interest.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory."
    )
    parser.add_argument(
        "-r",
        "--ref_mol",
        help="Reference molecule to be aligned to. If blank, first molecule in sdf file is used",
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
        ## Prepare mcss
        mcss = oechem.OEMCSSearch(oechem.OEMCSType_Approximate)
        atomexpr = oechem.OEExprOpts_DefaultAtoms
        bondexpr = oechem.OEExprOpts_DefaultBonds
        mcss.Init(ref, atomexpr, bondexpr)
        mcss.SetMCSFunc(oechem.OEMCSMaxBondsCompleteCycles())

        ## Prepare openeye aligned depiction
        alignres = oedepict.OEPrepareAlignedDepiction(mol, mcss)

        ## Write out image
        if alignres.IsValid():
            out_fn = (
                f"{os.path.join(args.output_dir, mol.GetTitle())}.png".replace(
                    " ", "_"
                )
            )
            disp = oedepict.OE2DMolDisplay(mol)
            clearbackground = False
            oedepict.OERenderMolecule(out_fn, disp, clearbackground)


if __name__ == "__main__":
    main()
