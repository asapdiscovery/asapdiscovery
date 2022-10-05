"""
Example usage:
    python fauxalysis_from_docking.py
        -c /data/chodera/paynea/posit_hybrid_no_relax_keep_water_filter_frag/mers_fauxalysis.csv
        -i /lila/data/chodera/kaminowb/stereochemistry_pred/mers/mers_fragalysis/posit_hybrid_no_relax_keep_water_filter
        -o /data/chodera/paynea/posit_hybrid_no_relax_keep_water_filter_frag
        -f /lila/data/chodera/kaminowb/stereochemistry_pred/fragalysis/aligned
"""
import sys, os, argparse, shutil, pickle as pkl, yaml
from openeye import oechem

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
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
        default="../data/cmpd_to_frag.yaml",
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

    return parser.parse_args()


def check_output(d):
    ## First check for result pickle file
    try:
        pkl.load(open(f"{d}/results.pkl", "rb"))
    except FileNotFoundError:
        return False

    ## Then check for other intermediate files
    du = oechem.OEDesignUnit()
    if not oechem.OEReadDesignUnit(f"{d}/predocked.oedu", du):
        return False

    if load_openeye_pdb(f"{d}/predocked.pdb").NumAtoms() == 0:
        return False

    if load_openeye_sdf(f"{d}/docked.sdf").NumAtoms() == 0:
        return False

    return True


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
        Of the form {('Compound_ID', dimer): 'Structure_Source'} where "dimer" is a boolean.
    """
    ## Set up SDF file that will hold all ligands
    all_ligs_monomer_ofs = oechem.oemolostream()
    all_ligs_monomer_ofs.SetFlavor(
        oechem.OEFormat_SDF, oechem.OEOFlavor_SDF_Default
    )
    all_ligs_dimer_ofs = oechem.oemolostream()
    all_ligs_dimer_ofs.SetFlavor(
        oechem.OEFormat_SDF, oechem.OEOFlavor_SDF_Default
    )
    # arranged this way so dimer=True => "dimer"
    dimers_strings = ["monomer", "dimer"]
    all_ofs = [all_ligs_monomer_ofs, all_ligs_dimer_ofs]
    for d, ofs in zip(dimers_strings, all_ofs):
        os.makedirs(f"{out_dir}", exist_ok=True)
        ofs.open(f"{out_dir}/combined.sdf")

    ## Loop through dict and parse input files into output files
    for (compound_id, dimer), best_str in best_structure_dict.items():
        ## Make sure input exists
        dimer_s = "dimer" if dimer else "monomer"
        compound_in_dir = f"{in_dir}/{compound_id}/{dimer_s}/{best_str}"
        compound_out_dir = f"{out_dir}/{compound_id}"
        if not check_output(compound_in_dir):
            print(
                (
                    f"No results found for {compound_id}/{dimer_s}/{best_str}, "
                    "skipping"
                ),
                flush=True,
            )
            continue
        elif os.path.exists(compound_out_dir):
            print(
                f"Fauxalysis directory exists at: "
                f"\t{compound_out_dir}"
                f"\tSkipping..."
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
        oechem.OEReadDesignUnit(f"{compound_in_dir}/predocked.oedu", du)
        prot = oechem.OEGraphMol()
        du.GetProtein(prot)
        lig = load_openeye_sdf(f"{compound_in_dir}/docked.sdf")

        ## Set ligand title
        lig.SetTitle(f"{compound_id}_{best_str}")

        ## First save apo
        save_openeye_pdb(prot, f"{compound_out_dir}/{best_str}_apo.pdb")

        ## Combine protein and ligand
        oechem.OEAddMols(prot, lig)

        save_openeye_pdb(prot, f"{compound_out_dir}/{compound_id}_bound.pdb")

        ## Remove Hs from lig and save SDF file
        for a in lig.GetAtoms():
            if a.GetAtomicNum() == 1:
                lig.DeleteAtom(a)
        save_openeye_sdf(lig, f"{compound_out_dir}/{compound_id}.sdf")
        ofs = all_ofs[int(dimer)]
        oechem.OEWriteMolecule(ofs, lig)

        if frag_dir:
            # TODO: Map the compound_id to the crystal structure and use that to build the path
            ## the compound ids used so far include a chain A vs chain B split, let's remove that

            if "_bound" in compound_id:
                frag_structure = compound_id
            else:
                compound_id_without_chain, chain = compound_id.split("_")
                frag_structure = (
                    f"{cmpd_to_frag_dict[compound_id_without_chain]}_{chain}"
                )

            frag_compound_path = os.path.join(frag_dir, frag_structure)
            bound_pdb_path = os.path.join(
                frag_compound_path, f"{frag_structure}_bound.pdb"
            )
            copied_bound_pdb_path = os.path.join(
                compound_out_dir, f"fragalysis_{compound_id}_bound.pdb"
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

    for ofs in all_ofs:
        ofs.close()


def main():
    args = get_args()

    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    docking_results = DockingResults(args.input_csv)

    best_structure_dict = {
        (values["Compound_ID"], values["Dimer"]): values["Structure_Source"]
        for values in docking_results.df.to_dict(orient="index").values()
    }

    ## Filter if directory already exists:
    best_structure_dict = {
        (cmpd, dimer): values
        for (cmpd, dimer), values in best_structure_dict.items()
        if not os.path.exists(f"{args.output_dir}/{cmpd}")
    }

    print(best_structure_dict)

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
