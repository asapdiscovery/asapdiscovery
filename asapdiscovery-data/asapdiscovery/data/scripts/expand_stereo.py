import argparse
import logging

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpansion
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander

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
    "--expand_defined",
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
        "stereo_expansion", path="./", stdout=True, level=logging.DEBUG
    )
    logger = logger_cls.getLogger()
    logger.info(f"Expanding stereoisomers for {args.infile} to {args.outfile}")
    logger.info(f"Forcing flip: {args.expand_defined}")
    logger.info(f"Debug: {args.debug}")

    infile = str(args.infile)
    ifs = oechem.oemolistream()
    if not ifs.open(infile):
        oechem.OEThrow.Fatal(f"Unable to open {infile} for reading")

    outfile = str(args.outfile)
    ofs = oechem.oemolostream()
    if not ofs.open(outfile):
        oechem.OEThrow.Fatal(f"Unable to open {outfile} for writing")

    ligs = []
    for mol in ifs.GetOEMols():
        ligs.append(Ligand.from_oemol(mol, compound_name="ExpansionMol"))

    ifs.close()

    expander = StereoExpander(
        input_ligands=ligs, stereo_expand_defined=args.expand_defined
    )

    expansions = expander.expand()
    expanded_ligs = StateExpansion.flatten_children(expansions)

    for lig in expanded_ligs:
        oechem.OEWriteMolecule(ofs, lig.to_oemol())

    ofs.close()


if __name__ == "__main__":
    main()
