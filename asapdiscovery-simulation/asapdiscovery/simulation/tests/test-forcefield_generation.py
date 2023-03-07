"""
@author: Alex Payne

Example usage:
python test-forcefield_generation.py \
-i /lila/data/chodera/asap-datasets/mers_fauxalysis/20230307_prepped_mers_pdbs/ \
-g "*_0.pdb"

python test-forcefield_generation.py \
-i "/Users/alexpayne/lilac-mount-point/asap-datasets/mers_fauxalysis/20230307_prepped_mers_pdbs/" \
-g "*_0.pdb"

# To run the basic test, just run without any arguments:
python test-forcefield_generation.py

"""
from asapdiscovery.simulation.utils import test_forcefield_generation
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml


def get_args():
    parser = argparse.ArgumentParser(description="")
    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_pdb_dir",
        default="inputs",
        type=str,
        help="Path to directory of pdb files.",
    )
    parser.add_argument(
        "-g",
        "--glob_string",
        default="*.pdb",
        type=str,
        help="String that selects the correct pdb files.",
    )
    parser.add_argument("-l", "--log_file", default="outputs/log.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    paths = [pdb_path for pdb_path in Path(args.input_pdb_dir).glob(args.glob_string)]
    error_dict = {}
    print(f"Preparing {len(paths)} structures")
    for path in tqdm(paths):
        try:
            test_forcefield_generation(str(path))
            outstr = "Success"
        except ValueError as error:
            outstr = error.__str__()
        error_dict[path.name] = outstr
    with open(args.log_file, "w") as f:
        yaml.safe_dump(error_dict, f)
