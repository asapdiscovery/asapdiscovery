import argparse
import logging

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import oechem, oeomega

parser = argparse.ArgumentParser(
    description="Enumerate steroisomers of molecules in a file"
)

parser.add_argument(
    "--infile",
    type=str,
    required=True,
    help="Path to the input file",
)

parser.add_argument(
    "--outfile",
    type=str,
    required=True,
    help="Path to the output file",
)

parser.add_argument(
    "--warts",
    action="store_true",
    help="Add warts to the output file",
)

parser.add_argument(
    "--force-flip",
    action="store_true",
    help="Force enumeration of stereo centers even if defined",
)

parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug messages",
)


def main():
    args = parser.parse_args()
    # setup logging
    logger_cls = FileLogger(
        "stereo_enumeration", path="./", stdout=True, level=logging.DEBUG
    )
    logger = logger_cls.getLogger()
    logger.info(f"Enumerating stereoisomers for {args.infile} to {args.outfile}")
    logger.info(f"Adding warts: {args.warts}")

    flipperOpts = oeomega.OEFlipperOptions()
    flipperOpts.SetWarts(args.warts)
    flipperOpts.SetEnumSpecifiedStereo(args.force_flip)

    ifs = oechem.oemolistream()
    if not ifs.open(args.infile):
        oechem.OEThrow.Fatal(f"Unable to open {args.infile} for reading")

    ofs = oechem.oemolostream()
    if not ofs.open(args.outfile):
        oechem.OEThrow.Fatal(f"Unable to open {args.outfile} for writing")

    for mol in ifs.GetOEMols():
        logger.info(f"Molecule Title: {mol.GetTitle()}")
        # set title to molecule name from postera if available
        mol_name_sd = oechem.OEGetSDData(mol.GetActive(), "Molecule Name")
        if mol_name_sd:
            logger.info(f"Molecule Name: {mol_name_sd}")
            mol.SetTitle(mol_name_sd)
        n_expanded = 0
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), flipperOpts):
            n_expanded += 1
            fmol = oechem.OEMol(enantiomer)
            oechem.OEWriteMolecule(ofs, fmol)
            if args.debug:
                # print smiles
                smiles = oechem.OEMolToSmiles(fmol)
                logger.debug(f"SMILES: {smiles}")
        logger.info(f"Expanded {n_expanded} stereoisomers")


if __name__ == "__main__":
    main()
