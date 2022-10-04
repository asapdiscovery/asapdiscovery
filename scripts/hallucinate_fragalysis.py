"""
Build library of ligands from a dataset of holo crystal structures docked to a
different dataset of apo structures.
"""
import argparse
from glob import glob
import itertools as it
import multiprocessing as mp
from openeye import oechem, oedocking, oespruce
import os
import pandas
import pickle as pkl
import re
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
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
from covid_moonshot_ml.docking.docking import (
    build_docking_system_direct,
    run_docking,
)
from covid_moonshot_ml.modeling import du_to_complex, make_du_from_new_lig


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


def mp_func(
    apo_prot,
    lig,
    ref_prot,
    out_dir,
    apo_name,
    lig_name,
    dimer,
    apo_chain,
    dock_sys,
    relax,
    loop_db,
    hybrid=False,
    save_du=False,
):
    out_base = f"{out_dir}/{apo_name}/"
    ## First check if this combo has already been run
    if check_output(out_base):
        dimer_s = "dimer" if dimer else "monomer"
        print(f"Results found for {lig_name}_{apo_name}_{dimer_s}", flush=True)
        return pkl.load(open(f"{out_base}/results.pkl", "rb"))

    ## Make output directory if necessary
    os.makedirs(out_base, exist_ok=True)
    out_fn = f"{out_base}/predocked"

    ## Make copy of lig so we don't modify original
    lig_copy = lig.CreateCopy()

    ## Make design unit and prep the receptor
    try:
        du = make_du_from_new_lig(
            apo_prot,
            lig_copy,
            dimer,
            ref_prot,
            False,
            False,
            "A",
            apo_chain,
            loop_db,
        )
    except AssertionError:
        print(
            f"Design unit generation failed for {lig_name}/{apo_name}",
            flush=True,
        )
        results = (lig_name, apo_name, dimer, None, -1.0, -1.0, -1.0)
        pkl.dump(results, open(f"{out_base}/results.pkl", "wb"))
        return results
    oedocking.OEMakeReceptor(du)

    ## Save if desired
    if save_du:
        oechem.OEWriteDesignUnit(f"{out_fn}.oedu", du)

    ## Get protein+lig complex in molecule form and save
    complex_mol = du_to_complex(du)
    save_openeye_pdb(complex_mol, f"{out_fn}.pdb")

    ## Keep track of if there's a clash (-1 if not using POSIT, 0 if no clash,
    ##  1 if there was a clash that couldn't be resolved)
    clash = -1

    ## Get ligand to dock
    dock_lig = oechem.OEMol()
    du.GetLigand(dock_lig)
    if dock_sys == "posit":
        ## Set up POSIT docking options
        opts = oedocking.OEPositOptions()
        ## kinoml has the below option set, but the accompanying comment implies
        ##  that we should be ignoring N stereochemistry, which, paradoxically,
        ##  corresponds to a False option (the default)
        opts.SetIgnoreNitrogenStereo(True)
        ## Set the POSIT methods to only be hybrid (otherwise leave as default
        ##  of all)
        if hybrid:
            opts.SetPositMethods(oedocking.OEPositMethod_HYBRID)

        ## Set up pose relaxation
        if relax == "clash":
            clash = 0
            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_CLASHED)
        elif relax == "all":
            clash = 0
            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
        elif relax != "none":
            ## Don't need to do anything for none bc that's already the default
            raise ValueError(f'Unknown arg for relaxation "{relax}"')

        print(
            f"Running POSIT {'hybrid' if hybrid else 'all'} docking with "
            f"{relax} relaxation for {lig_name}/{apo_name}",
            flush=True,
        )

        ## Set up poser object
        poser = oedocking.OEPosit(opts)
        poser.AddReceptor(du)

        ## Run posing
        pose_res = oedocking.OESinglePoseResult()
        ret_code = poser.Dock(pose_res, dock_lig)
    elif dock_sys == "hybrid":
        print("Running Hybrid docking", flush=True)

        ## Set up poser object
        poser = oedocking.OEHybrid()
        poser.Initialize(du)

        ## Run posing
        posed_mol = oechem.OEMol()
        ret_code = poser.DockMultiConformerMolecule(posed_mol, dock_lig)

        posit_prob = -1.0
    else:
        raise ValueError(f'Unknown docking system "{dock_sys}"')

    if ret_code == oedocking.OEDockingReturnCode_NoValidNonClashPoses:
        ## For POSIT with clash removal, if no non-clashing pose can be found,
        ##  re-run with no clash removal
        opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_NONE)
        clash = 1

        print(
            f"Re-running POSIT {'hybrid' if hybrid else 'all'} docking with "
            f"no relaxation for {lig_name}/{apo_name}",
            flush=True,
        )

        ## Set up poser object
        poser = oedocking.OEPosit(opts)
        poser.AddReceptor(du)

        ## Run posing
        pose_res = oedocking.OESinglePoseResult()
        ret_code = poser.Dock(pose_res, dock_lig)

    ## Check results
    if ret_code == oedocking.OEDockingReturnCode_Success:
        if dock_sys == "posit":
            posed_mol = pose_res.GetPose()
            posit_prob = pose_res.GetProbability()

        ## Get the Chemgauss4 score (adapted from kinoml)
        pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
        pose_scorer.Initialize(du)
        chemgauss_score = pose_scorer.ScoreLigand(posed_mol)
    else:
        err_type = oedocking.OEDockingReturnCodeGetName(ret_code)
        print(
            f"Pose generation failed for {lig_name}/{apo_name} ({err_type})",
            flush=True,
        )
        results = (lig_name, apo_name, dimer, None, -1.0, -1.0, -1.0)
        pkl.dump(results, open(f"{out_base}/results.pkl", "wb"))
        return results

    save_openeye_sdf(posed_mol, f"{out_base}/docked.sdf")
    save_openeye_sdf(dock_lig, f"{out_base}/predocked.sdf")

    ## Calculate RMSD
    oechem.OECanonicalOrderAtoms(dock_lig)
    oechem.OECanonicalOrderBonds(dock_lig)
    oechem.OECanonicalOrderAtoms(posed_mol)
    oechem.OECanonicalOrderBonds(posed_mol)
    ## Get coordinates, filtering out Hs
    predocked_coords = [
        c
        for a in dock_lig.GetAtoms()
        for c in dock_lig.GetCoords()[a.GetIdx()]
        if a.GetAtomicNum() != 1
    ]
    docked_coords = [
        c
        for a in posed_mol.GetAtoms()
        for c in posed_mol.GetCoords()[a.GetIdx()]
        if a.GetAtomicNum() != 1
    ]
    rmsd = oechem.OERMSD(
        oechem.OEDoubleArray(predocked_coords),
        oechem.OEDoubleArray(docked_coords),
        len(predocked_coords) // 3,
    )

    results = (
        lig_name,
        apo_name,
        dimer,
        f"{out_base}/docked.sdf",
        rmsd,
        posit_prob,
        chemgauss_score,
        clash,
    )
    pkl.dump(results, open(f"{out_base}/results.pkl", "wb"))
    return results


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments (these can be changed to eg yaml files later)
    parser.add_argument(
        "-apo",
        required=True,
        help="Wildcard string that will give all apo PDB files.",
    )
    parser.add_argument(
        "-holo",
        required=True,
        help="Wildcard string that will give all holo PDB files.",
    )
    parser.add_argument(
        "-ref",
        help=(
            "PDB file for reference structure to align all apo structure to "
            "before docking."
        ),
    )
    parser.add_argument("-loop", required=True, help="Spruce loop_db file.")
    parser.add_argument(
        "-x", help="Fragalysis crystal structure compound tracker CSV file."
    )
    parser.add_argument(
        "-s",
        "--smarts_queries",
        type=str,
        help="Path to csv file containing smarts queries.",
    )

    ## Performance arguments
    parser.add_argument(
        "-n", default=10, type=int, help="Number of processors to use."
    )
    parser.add_argument(
        "-sys",
        default="posit",
        help="Which docking system to use [posit, hybrid]. Defaults to posit.",
    )
    parser.add_argument(
        "-relax",
        default="none",
        help="When to run relaxation [none, clash, all]. Defaults to none.",
    )
    parser.add_argument(
        "-hybrid",
        action="store_true",
        help="Whether to only use hybrid docking protocol in POSIT.",
    )
    parser.add_argument(
        "-keep_wat",
        action="store_true",
        help="Keep water molecules in the Design Unit.",
    )

    ## Output arguments
    parser.add_argument("-o", required=True, help="Parent output directory.")
    parser.add_argument(
        "-du",
        action="store_true",
        help="Store intermediate OEDesignUnit objects.",
    )
    parser.add_argument(
        "-cache",
        help=(
            "Cache directory (will use .cache in "
            "output directory if not specified)."
        ),
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## Check -sys and -relax
    args.sys = args.sys.lower()
    args.relax = args.relax.lower()
    if args.sys not in {"posit", "hybrid"}:
        raise ValueError(f'Unknown docking system "{args.sys}"')
    if args.relax not in {"none", "clash", "all"}:
        raise ValueError(f'Unknown arg for relaxation "{args.relax}"')

    ## Set logging
    import logging

    logging.basicConfig(level=logging.DEBUG)

    ## Get all files and parse out a name
    all_apo_fns = glob(args.apo)
    all_apo_names = [
        os.path.splitext(os.path.basename(fn))[0] for fn in all_apo_fns
    ]
    all_holo_fns = glob(args.holo)
    all_holo_names = [
        os.path.splitext(os.path.basename(fn))[0] for fn in all_holo_fns
    ]

    if args.x is not None:
        ## Need to go up one level of directory
        frag_dir = os.path.dirname(os.path.dirname(args.holo))

        ## First, parse the fragalysis directory into a dictionary of
        ##  CrystalCompoundData
        sars_xtals = parse_fragalysis_data(args.x, frag_dir)

        ## Get dict mapping crystal structure id to compound id
        compound_id_dict = get_compound_id_xtal_dicts(sars_xtals.values())[1]

        ## Map all holo structure names to their ligand name
        all_holo_names = [
            f'{compound_id_dict[n.split("_")[0]]}_{n.split("_")[1]}'
            if n.split("_")[0] in compound_id_dict
            else n
            for n in all_holo_names
        ]

        if args.smarts_queries:
            ## For the compounds for which we have smiles strings, get a
            ##  dictionary mapping the Compound_ID to the smiles
            cmp_to_smiles_dict = {
                compound_id: data.smiles
                for compound_id, data in sars_xtals.items()
                if data.smiles
            }

            ## Filter based on the smiles using this OpenEye function
            filtered_inputs = filter_docking_inputs(
                smarts_queries=args.smarts_queries,
                docking_inputs=cmp_to_smiles_dict,
            )

            ## Keep track of which structures to keep
            keep_idx = [
                n.split("_")[0] in filtered_inputs for n in all_holo_names
            ]
        else:
            keep_idx = [True] * len(all_holo_names)

        ## Trim files and names to keep
        all_holo_fns = [fn for keep, fn in zip(keep_idx, all_holo_fns) if keep]
        all_holo_names = [
            n for keep, n in zip(keep_idx, all_holo_names) if keep
        ]

        ## Sanity check to make sure the lengths are the same
        assert len(all_holo_fns) == len(all_holo_names)

    print(f"{len(all_apo_fns)} apo structures")
    print(f"{len(all_holo_fns)} ligands to dock", flush=True)

    ## Get correct ligand chain for each file
    re_pat = r"Mpro-.*_[0-9]([AB])"
    all_matches = [re.search(re_pat, fn) for fn in all_holo_fns]
    ## Get rid of files that aren't A or B chain (can't handle that for now)
    lig_chains = [m.groups()[0] if m else None for m in all_matches]
    ## Get ligands from all holo structures
    all_ligs = [
        split_openeye_mol(load_openeye_pdb(fn), lig_chain=c)["lig"]
        for fn, c in zip(all_holo_fns, lig_chains)
        if c
    ]
    ## Trim names
    bad_holo_names = [
        n for i, n in enumerate(all_holo_names) if all_matches[i] is None
    ]
    all_holo_names = [n for i, n in enumerate(all_holo_names) if all_matches[i]]
    for n in bad_holo_names:
        print(f"Removed {n} (not A or B chain)", flush=True)

    ## Get proteins from apo structures
    all_prots = []
    for n, fn in zip(all_apo_names, all_apo_fns):
        split_dict = split_openeye_mol(load_openeye_pdb(fn))
        prot = split_dict["pro"]
        ## Add waters if required
        if args.keep_wat:
            oechem.OEAddMols(prot, split_dict["water"])

        ## Build monomer into dimer as necessary (will need to handle
        ##  re-labeling chains since the monomer seems to get the chainID C)
        ## Shouldn't affect the protein if the dimer has already been built
        bus = list(oespruce.OEExtractBioUnits(prot))
        ## Check to make sure everything got built correctly
        if len(bus) != 2:
            print(
                f"Incorrect number of Bio units built for {n} ({len(bus)})",
                flush=True,
            )
        if bus[0].NumAtoms() != 2 * bus[1].NumAtoms():
            print(
                (
                    f"Incorrect relative size of Bio units for {n} "
                    f"({bus[0].NumAtoms()} and {bus[1].NumAtoms()})"
                ),
                flush=True,
            )
        ## Need to cast to OEGraphMol bc returned type is OEMolBase, which
        ##  doesn't pickle
        all_prots.append(oechem.OEGraphMol(bus[0]))

    ## Parse reference
    if args.ref:
        ref_prot = split_openeye_mol(load_openeye_pdb(args.ref))["pro"]
    else:
        ref_prot = None

    ## Figure out cache dir for docking
    if args.cache is None:
        cache_dir = f"{args.o}/.cache/"
    else:
        cache_dir = args.cache
    os.makedirs(cache_dir, exist_ok=True)

    mp_args = []
    ## Construct all args for mp_func
    for lig_name, lig in zip(all_holo_names, all_ligs):
        for (dimer, apo_chain) in it.product([True, False], ["A", "B"]):
            dimer_s = "dimer" if dimer else "monomer"
            out_dir = f"{args.o}/{lig_name}/{dimer_s}"
            os.makedirs(out_dir, exist_ok=True)
            ## Load and parse apo protein
            for prot_name, apo_prot in zip(all_apo_names, all_prots):
                mp_args.append(
                    (
                        apo_prot,
                        lig,
                        ref_prot,
                        out_dir,
                        f"{prot_name}_{apo_chain}",
                        lig_name,
                        dimer,
                        apo_chain,
                        args.sys,
                        args.relax,
                        args.loop,
                        args.hybrid,
                        args.du,
                    )
                )

    results_cols = [
        "SARS_ligand",
        "MERS_structure",
        "dimer",
        "docked_file",
        "docked_RMSD",
        "POSIT_prob",
        "chemgauss4_score",
        "clash",
    ]
    nprocs = min(mp.cpu_count(), len(mp_args), args.n)
    print(f"Running {len(mp_args)} docking runs over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        results_df = pool.starmap(mp_func, mp_args)
    results_df = pandas.DataFrame(results_df, columns=results_cols)

    results_df.to_csv(f"{args.o}/all_results.csv")

    ## Will probably at some point want to integrate this into the actual
    ##  function instead of having it run afterwards
    # write_fragalysis_output(args.o, f'{args.o.rstrip("/")}_frag/')


if __name__ == "__main__":
    main()
