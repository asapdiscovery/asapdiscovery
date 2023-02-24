import argparse

from asapdiscovery.data.utils import cdd_to_schema  # noqa: E402
from asapdiscovery.data.utils import cdd_to_schema_pair  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-i", required=True, help="CSV file input from CDD.")
    parser.add_argument("-json", required=True, help="Output JSON file.")
    parser.add_argument("-csv", help="Output CSV file.")
    parser.add_argument(
        "-type",
        default="std",
        help=(
            "What type of data is " "being loaded (std: standard, ep: enantiomer pairs)"
        ),
    )
    parser.add_argument(
        "-achiral", action="store_true", help="Remove chiral molecules."
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.type.lower() == "std":
        _ = cdd_to_schema(args.i, args.json, args.csv, args.achiral)
    elif args.type.lower() == "ep":
        _ = cdd_to_schema_pair(args.i, args.json, args.csv)
    else:
        raise ValueError(f"Unknown value for -type: {args.type}.")


if __name__ == "__main__":
    main()
