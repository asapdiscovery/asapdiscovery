import sys, os, argparse, shutil, pandas
from openeye import oechem

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from covid_moonshot_ml.docking.analysis import DockingResults
from covid_moonshot_ml.datasets.utils import (
    get_compound_id_xtal_dicts,
    load_openeye_pdb,
    load_openeye_sdf,
    parse_fragalysis_data,
    save_openeye_pdb,
    save_openeye_sdf,
    split_openeye_mol,
    filter_docking_inputs,
)


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_csv",
        required=True,
        help="Path to CSV file containing best results.",
    )
    parser.add_argument(
        "-f",
        "--fragalysis_dir",
        default=False,
        help="Path to fragalysis results directory.",
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


def write_fragalysis_output(in_dir, out_dir, sort_key="POSIT_prob"):
    """
    Convert original-style output structure to a fragalysis-style output
    structure.

    Parameters
    ----------
    in_dir : str
        Top-level directory of original-style output
    out_dir : str
        Top-level directory of new output
    sort_key : str, default="POSIT_prob"
        Which column to use to pick the best structure for each compound.
        Options are [docked_RMSD, POSIT_prob, chemgauss4_score]. If POSIT_prob
        is selected but not present in CSV file, fall back to docked_RMSD, then
        chemgauss4_score if all docked_RMSD are -1.
    """
    ## Load master results CSV file
    df = pandas.read_csv(f"{in_dir}/all_results.csv", index_col=0)

    ## Set up test for best structure
    if (sort_key == "POSIT_prob") and ("POSIT_prob" not in df.columns):
        print(
            "POSIT_prob not in given CSV file, falling back to docked_RMSD",
            flush=True,
        )
        sort_key = "docked_RMSD"
    if sort_key == "POSIT_prob":
        sort_fn = lambda s: s.idxmax()
    elif (sort_key == "docked_RMSD") or (sort_key == "chemgauss4_score"):
        sort_fn = lambda s: s.idxmin()
    else:
        raise ValueError(f'Unknown sort_key "{sort_key}"')

    ## Get best structure for each
    best_str_dict = {}
    for ((compound_id, dimer), g) in df.groupby(["SARS_ligand", "dimer"]):
        if (sort_key == "docked_RMSD") and (g["docked_RMSD"] == -1.0).all():
            sort_key = "chemgauss4_score"
        best_str_dict[(compound_id, dimer)] = g.loc[
            sort_fn(g[sort_key]), f"MERS_structure"
        ]

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
        os.makedirs(f"{out_dir}/{d}", exist_ok=True)
        ofs.open(f"{out_dir}/{d}/combined.sdf")

    ## Loop through dict and parse input files into output files
    for (compound_id, dimer), best_str in best_str_dict.items():
        ## Make sure input exists
        dimer_s = "dimer" if dimer else "monomer"
        tmp_in_dir = f"{in_dir}/{compound_id}/{dimer_s}/{best_str}"
        if not check_output(tmp_in_dir):
            print(
                (
                    f"No results found for {compound_id}/{dimer_s}/{best_str}, "
                    "skipping"
                ),
                flush=True,
            )
        ## Create output directory if necessary
        tmp_out_dir = f"{out_dir}/{dimer_s}/{compound_id}"
        os.makedirs(tmp_out_dir, exist_ok=True)

        ## Load necessary files
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(f"{tmp_in_dir}/predocked.oedu", du)
        prot = oechem.OEGraphMol()
        du.GetProtein(prot)
        lig = load_openeye_sdf(f"{tmp_in_dir}/docked.sdf")

        ## Set ligand title
        lig.SetTitle(f"{compound_id}_{best_str}")

        ## First save apo
        save_openeye_pdb(prot, f"{tmp_out_dir}/{best_str}_apo.pdb")

        ## Combine protein and ligand
        oechem.OEAddMols(prot, lig)

        # ## Get number of residues per chain
        # num_res_chain_dict = {}
        # hv = oechem.OEHierView(prot)
        # for chain in hv.GetChains():
        #     max_resid = 0
        #     for frag in chain.GetFragments():
        #         frag_max = max(
        #             [r.GetResidueNumber() for r in frag.GetResidues()]
        #         )
        #         if frag_max > max_resid:
        #             max_resid = frag_max
        #     num_res_chain_dict[chain.GetChainID()] = max_resid
        # ## Rename and renumber ligand residue
        # for a in prot.GetAtoms():
        #     r = oechem.OEAtomGetResidue(a)
        #     if r.GetName() == "UNL":
        #         r.SetHetAtom(True)
        #         r.SetName("LIG")
        #         r.SetResidueNumber(num_res_chain_dict[r.GetChainID()] + 100)
        #     oechem.OEAtomSetResidue(a, r)

        save_openeye_pdb(prot, f"{tmp_out_dir}/{compound_id}_bound.pdb")

        ## Remove Hs from lig and save SDF file
        for a in lig.GetAtoms():
            if a.GetAtomicNum() == 1:
                lig.DeleteAtom(a)
        save_openeye_sdf(lig, f"{tmp_out_dir}/{compound_id}.sdf")
        ofs = all_ofs[int(dimer)]
        oechem.OEWriteMolecule(ofs, lig)

    for ofs in all_ofs:
        ofs.close()


def main():
    args = get_args()

    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    docking_results = DockingResults(args.input_csv)

    for index, values in docking_results.df.to_dict(orient="index").items():
        input_dir_path = os.path.dirname(values["Docked_File"])
        output_dir_path = os.path.join(args.output_dir, values["Complex_ID"])

        if os.path.exists(input_dir_path) and not os.path.exists(
            output_dir_path
        ):
            # print(input_dir_path)
            # print(output_dir_path)
            shutil.copytree(input_dir_path, output_dir_path)

        if args.fragalysis_dir:
            compound_id = values["Compound_ID"]
            compound_path = os.path.join(args.fragalysis_dir, compound_id)
            bound_pdb_path = os.path.join(
                compound_path, f"{compound_id}_bound.pdb"
            )
            if os.path.exists(bound_pdb_path):
                print(bound_pdb_path)
                shutil.copy2(bound_pdb_path, output_dir_path)

    ## Will probably at some point want to integrate this into the actual
    ##  function instead of having it run afterwards
    write_fragalysis_output(args.o, f'{args.o.rstrip("/")}_frag/')


if __name__ == "__main__":
    main()
