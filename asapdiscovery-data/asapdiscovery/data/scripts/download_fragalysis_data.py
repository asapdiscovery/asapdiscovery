import argparse

from asapdiscovery.data.fragalysis import (  # noqa: E402
    API_CALL_BASE,
    download,
)


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

    # Overwrite the base target with the cli-specified target
    API_CALL_BASE["target_name"] = args.target.capitalize()

    download(args.o, API_CALL_BASE, args.x)


if __name__ == "__main__":
    main()
