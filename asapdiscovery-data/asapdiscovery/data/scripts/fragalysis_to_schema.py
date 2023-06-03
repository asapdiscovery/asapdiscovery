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
    parser.add_argument(
        "--name_filter",
        type=str,
        default=None,
        help="Filter to apply to Fragalysis names (i.e. Mpro-P for only P structures).",
    )
    parser.add_argument(
        "--name_filter_column",
        type=str,
        default="crystal_name",
        help="Column of metadata csv to apply name filter to.",
    )
    parser.add_argument(
        "--drop_duplicates",
        default=False,
        action="store_true",
        help="Drop structures with the same 'RealCrystalName'.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    check_filelist_has_elements([args.metadata_csv, args.aligned_dir])
    xtals = parse_fragalysis(
        args.metadata_csv,
        args.aligned_dir,
        name_filter=args.name_filter,
        name_filter_column=args.name_filter_column,
        drop_duplicate_datasets=args.drop_duplicates,
    )
    CrystalCompoundDataset.from_list(xtals).to_csv(args.output_dir / "fragalysis.csv")
