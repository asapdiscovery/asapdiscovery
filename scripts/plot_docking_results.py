import sys, os, argparse
sys.path.append(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
from covid_moonshot_ml.docking.analysis import DockingDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-i', "--input_csv", required=True,
        help='CSV file containing docking results.')

    return parser.parse_args()

def main():
    args = get_args()

    pkl_fn = os.path.join(args.d, 'mcs_sort_index.pkl')
    dd = DockingDataset(pkl_fn=pkl_fn, dir_path=args.d)
    dd.read_pkl()
    dd.analyze_docking_results(args.f,
                               args.c,
                               test=False)

if __name__ == '__main__':
    main()