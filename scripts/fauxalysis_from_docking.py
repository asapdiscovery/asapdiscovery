"""
After having generated some docking results, this script takes a CSV file containing the best structures for each
Compound_ID and generates a fragalysis-like "fauxalysis" dataset. This includes copying over the original structure
of the compound bound to SARS-2 Mpro from Fragalysis.

Example usage:
    python fauxalysis_from_docking.py
        -c /data/chodera/paynea/posit_hybrid_no_relax_keep_water_filter_frag/mers_fauxalysis.csv
        -i /lila/data/chodera/kaminowb/stereochemistry_pred/mers/mers_fragalysis/posit_hybrid_no_relax_keep_water_filter
        -o /data/chodera/paynea/posit_hybrid_no_relax_keep_water_filter_frag
        -f /lila/data/chodera/kaminowb/stereochemistry_pred/fragalysis/aligned
"""
import sys, os, argparse, shutil, pickle as pkl, yaml, re
from openeye import oechem

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from covid_moonshot_ml.docking.analysis import DockingResults
from covid_moonshot_ml.datasets.utils import (
    load_openeye_pdb,
    load_openeye_sdf,
    save_openeye_pdb,
    save_openeye_sdf,
)


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-c",
        "--input_csv",
        required=True,
        help="Path to CSV file containing best results.",
    )
    parser.add_argument(
        "-f",
        "--fragalysis_dir",
        default=None,
        help="Path to fragalysis results directory.",
    )
    parser.add_argument(
        "-y",
        "--fragalysis_yaml",
        default=os.path.join(
            repo_path,
            "data",
            "cmpd_to_frag.yaml",
        ),
        help="Path to yaml file containing a compound-to-fragalysis dictionary.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to directory containing all the docking results.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to newly created fauxalysis directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag to enable overwriting output data, otherwise it will skip directories that exists already.",
    )
    parser.add_argument(
        "-p",
        "--prepped_path",
        required=True,
        help="Path to prepped receptors.",
    )

    return parser.parse_args()


def check_output(d):
    ## First check for result pickle file
    try:
        pkl.load(open(f"{d}/results.pkl", "rb"))
    except FileNotFoundError:
        return False

    ## Then check for other intermediate files
    # du = oechem.OEDesignUnit()
    # if not oechem.OEReadDesignUnit(f"{d}/predocked.oedu", du):
    #     return False
    #
    # if load_openeye_pdb(f"{d}/predocked.pdb").NumAtoms() == 0:
    #     return False

    if load_openeye_sdf(f"{d}/docked.sdf").NumAtoms() == 0:
        return False

    return True


# ToDo: Move this as well as other scripts to a more logical api
def write_fragalysis_output(
    in_dir, out_dir, best_structure_dict, frag_dir=None, cmpd_to_frag_dict=None
):
    """
    Convert original-style output structure to a fragalysis-style output
    structure.

    Parameters
    ----------
    in_dir : str
        Top-level directory of original-style output
    out_dir : str
        Top-level directory of new output
    frag_dir : str
        Path to fragalysis results directory.
    best_structure_dict : dict
        Of the form {'Complex_ID': {key: value}}
    """
    os.makedirs(f"{out_dir}", exist_ok=True)

    ## Create list of sdf files we will make so that we can concatenate them at the end
    cmpd_sdf_list = []

    ## Loop through dict and parse input files into output files
    for complex_id, complex_dict in best_structure_dict.items():
        # docked_sdf = complex_dict.get("Docked_File")
        receptor_oedu = complex_dict.get("Prepped_Receptor")
        ## Make sure input exists
        # dimer_s = "dimer" if dimer else "monomer"
        # compound_in_dir = os.path.dirname(docked_sdf)
        compound_in_dir = os.path.join(in_dir, complex_id)
        compound_out_dir = os.path.join(out_dir, complex_id)

        ## If inputs don't exist, else if the output directory already exists, don't waste time
        if not check_output(compound_in_dir):
            print(
                (f"No results found for {compound_in_dir}, " "skipping"),
                flush=True,
            )
            continue
        else:
            print(
                (
                    f"Generating fauxalysis...\n"
                    f"\tfrom: {compound_in_dir}\n"
                    f"\tto: {compound_out_dir}"
                ),
                flush=True,
            )

        ## Create output directory if necessary
        os.makedirs(compound_out_dir, exist_ok=True)

        ## Load necessary files
        du = oechem.OEDesignUnit()
        # oechem.OEReadDesignUnit(f"{compound_in_dir}/predocked.oedu", du)
        oechem.OEReadDesignUnit(receptor_oedu, du)
        prot = oechem.OEGraphMol()
        du.GetProtein(prot)
        lig = load_openeye_sdf(f"{compound_in_dir}/docked.sdf")

        ## Set ligand title, useful for the combined sdf file
        lig.SetTitle(f"{complex_id}")

        ## Give ligand atoms their own chain "L" and set the resname to "LIG"
        residue = oechem.OEAtomGetResidue(next(iter(lig.GetAtoms())))
        residue.SetChainID("L")
        residue.SetName("LIG")
        residue.SetHetAtom(True)
        for atom in list(lig.GetAtoms()):
            oechem.OEAtomSetResidue(atom, residue)

        ## First save apo
        save_openeye_pdb(prot, f"{compound_out_dir}/{complex_id}_apo.pdb")

        ## Combine protein and ligand and save
        ## TODO: consider saving water as well
        oechem.OEAddMols(prot, lig)

        ## Clean up PDB info by re-perceiving, perserving chain ID, residue number, and residue name
        preserve = (
            oechem.OEPreserveResInfo_ChainID
            | oechem.OEPreserveResInfo_ResidueNumber
            | oechem.OEPreserveResInfo_ResidueName
        )
        oechem.OEPerceiveResidues(prot, preserve)

        save_openeye_pdb(prot, f"{compound_out_dir}/{complex_id}_bound.pdb")

        ## Save first to its own sdf file
        cmpd_sdf = f"{compound_out_dir}/{complex_id}.sdf"
        cmpd_sdf_list.append(cmpd_sdf)
        save_openeye_sdf(lig, cmpd_sdf)

        ## Copy over file from original fragalysis directory
        if frag_dir:
            compound_id_list = compound_id.split("_")
            compound_id_without_chain = compound_id_list[0]
            chain = compound_id_list[1]

            ## If the compound_id has "_bound" in it, it was constructed directly from the fragalysis dataset name
            ## If not, then we have to get the fragalysis dataset name from the cmpd_to_frag_dict
            if "_bound" in compound_id:
                ## The way this code works, basically it drops the "_bound" suffix from the original compound name
                frag_structure = f"{compound_id_without_chain}_{chain}"
            else:
                frag_structure = (
                    f"{cmpd_to_frag_dict[compound_id_without_chain]}_{chain}"
                )

            frag_compound_path = os.path.join(frag_dir, frag_structure)
            bound_pdb_path = os.path.join(
                frag_compound_path, f"{frag_structure}_bound.pdb"
            )
            copied_bound_pdb_path = os.path.join(
                compound_out_dir, f"fragalysis_{frag_structure}_bound.pdb"
            )
            if os.path.exists(bound_pdb_path):
                print(
                    f"Copying fragalysis source\n"
                    f"\tfrom {bound_pdb_path}\n"
                    f"\tto {copied_bound_pdb_path}"
                )
                shutil.copy2(
                    bound_pdb_path,
                    copied_bound_pdb_path,
                )
            else:
                print(f"Fragalysis source not found:\n" f"\t{bound_pdb_path}")

    ## Use shutil again to concatenate all the sdf files into one combined file!
    ## I did this because I was seeing errors in the combined.sdf when created using OpenEye
    combined_sdf = f"{out_dir}/combined.sdf"

    with open(combined_sdf, "wb") as wfd:
        for f in cmpd_sdf_list:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)


def main():
    args = get_args()
    print(args.fragalysis_yaml)

    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    docking_results = DockingResults(args.input_csv)

    best_structure_dict_all = {
        values["Complex_ID"]: values
        for values in docking_results.df.to_dict(orient="index").values()
    }

    ## Get Prepped_Receptor
    prepped_dir_list = os.listdir(args.prepped_path)
    for complex_id, values in best_structure_dict_all.items():
        dirname = [
            dirname
            for dirname in prepped_dir_list
            if values["Structure_Source"] in dirname
        ][0]
        values["Prepped_Receptor"] = os.path.join(
            args.prepped_path,
            dirname,
            "prepped_receptor.oedu",
        )
        best_structure_dict_all[complex_id] = values

    if args.overwrite:
        best_structure_dict = best_structure_dict_all
    else:
        # Filter if directory already exists:
        best_structure_dict = {}
        for complex_id, complex_dict in best_structure_dict_all.items():
            if not os.path.exists(f"{args.output_dir}/{complex_id}"):
                best_structure_dict[complex_id] = complex_dict
            else:
                print(
                    f"Skipping {complex_id} since output already exists at:\n"
                    f"\t{args.output_dir}/{complex_id}"
                )

    ## Get cmpd_to_fragalysis source dict if required
    if args.fragalysis_dir:
        with open(args.fragalysis_yaml, "r") as f:
            cmpd_to_frag_dict = yaml.safe_load(f)
    else:
        cmpd_to_frag_dict = None
    write_fragalysis_output(
        args.input_dir,
        args.output_dir,
        best_structure_dict,
        args.fragalysis_dir,
        cmpd_to_frag_dict,
    )


if __name__ == "__main__":
    main()
