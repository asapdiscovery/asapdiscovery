import argparse
import copy

from asapdiscovery.data.fragalysis import API_CALL_BASE, download  # noqa: E402


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

    # Copy the base call and update the base target with the cli-specified target
    api_call = copy.deepcopy(API_CALL_BASE)
    api_call["target_name"] = args.target.capitalize()

    download(args.o, api_call, args.x)


if __name__ == "__main__":
    main()
