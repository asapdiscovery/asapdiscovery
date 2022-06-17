import argparse
import sys

sys.path.append('.')
from covid_moonshot_ml.datasets.moonshot import download_achiral

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-tok', required=True,
        help='File containing CDD token.')
    parser.add_argument('-o', required=True, help='Output CSV file.')

    return(parser.parse_args())

def main():
    args = get_args()

    header = {'X-CDD-token': ''.join(open(args.tok, 'r').readlines()).strip()}
    _ = download_achiral(header, fn_out=args.o)

if __name__ == '__main__':
    main()