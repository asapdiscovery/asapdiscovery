import sys, os, argparse, shutil

sys.path.append(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
from covid_moonshot_ml.docking.analysis import DockingResults


def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-i', "--input_csv", required=True,
                        help='Path to CSV file containing best results.')
    parser.add_argument('-o', "--output_dir", required=True,
                        help="Path to newly created fauxalysis directory")

    return parser.parse_args()

def main():
    args = get_args()

    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    docking_results = DockingResults(args.input_csv)

    for index, values in docking_results.df.to_dict(orient='index').items():
        print(values)
        input_dir_path = os.path.dirname(values["Docked_File"])
        print(input_dir_path)
        output_dir_path = os.path.join(args.output_dir, values["Complex_ID"])
        shutil.copy2(input_dir_path, output_dir_path)

if __name__ == '__main__':
    main()