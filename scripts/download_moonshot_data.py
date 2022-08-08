"""
Download Moonshot data

"""

import argparse
import os
import sys

import logging
#logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from covid_moonshot_ml.datasets.moonshot import download_molecules

################################################################################
def get_args():
    # TODO: Update to more modern, full-featured command-line
    # e.g. click : https://click.palletsprojects.com/
    parser = argparse.ArgumentParser(description='')

    # TODO: Enable token to be specified by environment variable
    parser.add_argument('-tok', required=True,
        help='File containing CDD token.')

    # TODO: Include options for specifying which subsets to include (achiral, racemic, enantiopure)
    # TODO: Include option for specifying which SMILES field to use

    parser.add_argument('-o', required=True, help='Output CSV file.')

    return(parser.parse_args())

def main():
    args = get_args()

    header = {'X-CDD-token': ''.join(open(args.tok, 'r').readlines()).strip()}
    #_ = download_achiral(header, fn_out=args.o)
    _ = download_molecules(header, smiles_fieldname='suspected_SMILES', retain_achiral=True, retain_racemic=True, fn_out=args.o)

if __name__ == '__main__':
    main()
