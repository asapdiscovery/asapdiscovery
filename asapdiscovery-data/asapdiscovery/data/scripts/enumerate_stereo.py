import argparse
import logging

from asapdiscovery.data.expand_stereo import StereoExpander, StereoExpanderOptions
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
    logger.info(f"Forcing flip: {args.force_flip}")
    logger.info(f"Debug: {args.debug}")

    # setup options
    options = StereoExpanderOptions(
        warts=args.warts,
        force_flip=args.force_flip,
        debug=args.debug,
        postera_names=True,
    )

    # setup expander
    expander = StereoExpander(options, logger=logger)
    expander.expand_structure_file(args.infile, args.outfile)


if __name__ == "__main__":
    main()
