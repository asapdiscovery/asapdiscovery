import argparse

from asapdiscovery.data.fragalysis import (
    download,
    MAC1_API_CALL,
    MPRO_API_CALL,
)  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Which target to download. Options are [mpro, mac1].",
    )
    parser.add_argument("-o", required=True, help="Output file name.")
    parser.add_argument(
        "-x", action="store_true", help="Extract file after downloading it."
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.target.lower() == "mpro":
        api_call = MPRO_API_CALL
    elif args.target.lower() == "mac1":
        api_call = MAC1_API_CALL
    else:
        raise ValueError(f"Unknown target: {args.target}")

    download(args.o, api_call, args.x)


if __name__ == "__main__":
    main()
