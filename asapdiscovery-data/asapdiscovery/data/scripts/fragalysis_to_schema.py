import argparse
from pathlib import Path
from asapdiscovery.data.utils import check_filelist_has_elements
from asapdiscovery.data.schema import CrystalCompoundDataset
from asapdiscovery.data.fragalysis import parse_fragalysis


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--metadata_csv",
        required=True,
        type=Path,
        help="Input metadata CSV file.",
    )
    parser.add_argument(
        "--aligned_dir",
        required=True,
        type=Path,
        help="Path to directory called 'aligned' in fragalysis download.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=Path,
        help="Path to output directory.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    check_filelist_has_elements([args.metadata_csv, args.aligned_dir])
    xtals = parse_fragalysis(args.metadata_csv, args.aligned_dir)
    CrystalCompoundDataset.from_list(xtals).to_csv(args.output_dir / "fragalysis.csv")
