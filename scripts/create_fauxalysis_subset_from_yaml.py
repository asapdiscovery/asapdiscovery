"""
The purpose of this script is to move a subset of the fauxalysis directory into a new directory.
The subsetting is done based on the names of PDB files contained in a yaml file passed to the script.
It also handles getting a new combined SDF file for only the complexes in selection
Example Usage:
    python create_fauxalysis_subset_from_yaml.py
        -d /Volumes/Rohirrim/local_test/mers_hallucination_hybrid/posit_hybrid_no_relax_keep_water_frag
        -o /Volumes/Rohirrim/local_test/mers_hallucination_hybrid/posit_hybrid_no_relax_keep_water_frag_carlsson

"""
import sys, os, argparse, yaml, shutil

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d", "--fauxalysis_dir", required=True, help="Path to fauxalysis_dir."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output dir"
    )
    parser.add_argument(
        "-y",
        "--yaml_file",
        default=os.path.join(
            repo_path,
            "data",
            "luttens2022ultralarge.yaml",
        ),
        help="Path to yaml_file",
    )

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.yaml_file, "r") as f:
        pdb_dict = yaml.safe_load(f)

    ## Get a list of directories and copy over all non-directory files
    complex_dir_list = []
    for item in os.listdir(args.fauxalysis_dir):
        full_path = os.path.join(args.fauxalysis_dir, item)
        if os.path.isdir(full_path):
            complex_dir_list.append(item)
        else:
            in_file = os.path.join(args.fauxalysis_dir, full_path)
            print(f"Copying {in_file}")
            shutil.copy2(in_file, args.output_dir)

    sdf_list = []
    for flag in pdb_dict.keys():

        ## if our flag of interest is in any complex, include it!
        ## TODO: replace with REGEX?
        complexes = [
            complex_id
            for complex_id in complex_dir_list
            if flag.lower() in complex_id
        ]

        ## Copy files over using shutil
        for complex in complexes:
            in_path = os.path.join(args.fauxalysis_dir, complex)
            sdf_list.append(os.path.join(in_path, f"{complex}.sdf"))
            out_path = os.path.join(args.output_dir, complex)
            try:
                shutil.copytree(in_path, out_path)
                print(
                    f"Copying...\n" f"\tfrom: {in_path}" f"\t  to: {out_path}"
                )
            except FileExistsError:
                print(
                    f"File exists, skipping!\n"
                    f"\tfrom: {in_path}"
                    f"\t  to: {out_path}"
                )
                pass

    ## Use shutil again to concatenate all the sdf files into one combined file!
    combined_sdf = f"{args.output_dir}/combined.sdf"
    print(sdf_list)

    with open(combined_sdf, "wb") as wfd:
        for f in sdf_list:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)


if __name__ == "__main__":
    main()
