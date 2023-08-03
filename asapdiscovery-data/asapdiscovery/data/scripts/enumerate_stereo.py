import argparse
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.logging import FileLogger


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


def main():
    args = parser.parse_args()
    # setup logging
    logger_cls = FileLogger(
        "stereo_enumeration", path="./", stdout=True, level=logging.DEBUG
    )
    logger = logger_cls.getLogger()
    logger.info(f"Enumerating stereoisomers for {args.infile} to {args.outfile}")
    logger.info(f"Adding warts: {args.warts}")

    flipperOpts = oeomega.OEFlipperOptions().SetWarts(args.warts)

    ifs = oechem.oemolistream()
    if not ifs.open(infile):
        oechem.OEThrow.Fatal(f"Unable to open {infile} for reading")

    ofs = oechem.oemolostream()
    if not ofs.open(outfile):
        oechem.OEThrow.Fatal(f"Unable to open {outfile} for writing")

    for mol in ifs.GetOEMols():
        logger.info(f"Molecule: {mol.GetTitle()}")
        n_expanded = 0
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), flipperOpts):
            n_expanded += 1
            fmol = oechem.OEMol(enantiomer)
            oechem.OEWriteMolecule(ofs, fmol)
        logger.info(f"Expanded {n_expanded} stereoisomers")


if __name__ == "__main__":
    main()
