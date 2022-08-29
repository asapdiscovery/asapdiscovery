"""
Build library of ligands from a dataset of holo crystal structures docked to a
different dataset of apo structures.
"""
import argparse
from glob import glob
import multiprocessing as mp
from openeye import oechem, oedocking
import os
import pandas
import pickle as pkl
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
)
from covid_moonshot_ml.docking.docking import (
    build_docking_system_direct,
    run_docking,
)
from covid_moonshot_ml.modeling import du_to_complex, make_du_from_new_lig


def add_compound_id(in_fn, frag_xtal_fn, frag_xtal_dir, out_fn=None):
    compound_id_dict = get_compound_id_xtal_dicts(
        parse_fragalysis_data(frag_xtal_fn, frag_xtal_dir).values()
    )[1]

    ## Load original csv file
    df = pandas.read_csv(in_fn, index_col=0)

    ## Add column for compound_id
    df["SARS_compound_id"] = [
        compound_id_dict[xtal.split("_")[0]]
        if xtal.split("_")[0] in compound_id_dict
        else None
        for xtal in df["SARS_structure"]
    ]

    ## Reorder columns
    new_cols = [
        "MERS_structure",
        "SARS_structure",
        "SARS_compound_id",
        "docked_file",
        "docked_RMSD",
        "POSIT_prob",
        "chemgauss4_score",
    ]
    df = df.reindex(columns=new_cols)

    ## Save output
    if out_fn is not None:
        df.to_csv(out_fn)

    return df


def check_output(d):
    ## First check for result pickle file
    try:
        pkl.load(open(f"{d}/results.pkl", "rb"))
    except Exception:
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


def mp_func(
    apo_prot, lig, ref_prot, out_dir, apo_name, lig_name, save_du=False
):

    out_base = f"{out_dir}/{apo_name}/"
    ## First check if this combo has already been run
    if check_output(out_base):
        print(f"Results found for {lig_name}_{apo_name}", flush=True)
        return pkl.load(open(f"{out_base}/results.pkl", "rb"))

    ## Make output directory if necessary
    os.makedirs(out_base, exist_ok=True)
    out_fn = f"{out_base}/predocked"

    ## Make copy of lig so we don't modify original
    lig_copy = lig.CreateCopy()

    ## Make design unit and prep the receptor
    try:
        du = make_du_from_new_lig(apo_prot, lig_copy, ref_prot, False, False)
    except AssertionError:
        print(f"Design unit generation failed for {lig_name}/{apo_name}")
        results = (lig_name, apo_name, None, -1.0, -1.0, -1.0)
        pkl.dump(results, open(f"{out_base}/results.pkl", "wb"))
        return results
    oedocking.OEMakeReceptor(du)

    ## Save if desired
    if save_du:
        oechem.OEWriteDesignUnit(f"{out_fn}.oedu", du)

    ## Get protein+lig complex in molecule form and save
    complex_mol = du_to_complex(du)
    save_openeye_pdb(complex_mol, f"{out_fn}.pdb")

    ## Set up POSIT docking options
    opts = oedocking.OEPositOptions()
    ## kinoml has the below option set, but the accompanying comment implies
    ##  that we should be ignoring N stereochemistry, which, paradoxically,
    ##  corresponds to a False option (the default)
    # opts.SetIgnoreNitrogenStereo(True)
    opts.SetPositMethods(
        oedocking.OEPositMethod_FRED
        | oedocking.OEPositMethod_HYBRID
        | oedocking.OEPositMethod_SHAPEFIT
    )

    ## Set up poser object
    poser = oedocking.OEPosit(opts)
    poser.AddReceptor(du)

    ## Run posing
    dock_lig = oechem.OEMol()
    du.GetLigand(dock_lig)
    pose_res = oedocking.OESinglePoseResult()
    ret_code = poser.Dock(pose_res, dock_lig)

    ## Check results
    if ret_code == oedocking.OEDockingReturnCode_Success:
        posed_mol = pose_res.GetPose()
        posit_prob = pose_res.GetProbability()

        ## Get the Chemgauss4 score (adapted from kinoml)
        pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
        pose_scorer.Initialize(du)
        chemgauss_score = pose_scorer.ScoreLigand(posed_mol)
    else:
        print(f"Pose generation failed for {lig_name}/{apo_name}")
        results = (lig_name, apo_name, None, -1.0, -1.0, -1.0)
        pkl.dump(results, open(f"{out_base}/results.pkl", "wb"))
        return results

    save_openeye_sdf(posed_mol, f"{out_base}/docked.sdf")

    ## Need to remove Hs for RMSD calc
    docked_copy = posed_mol.CreateCopy()
    for a in docked_copy.GetAtoms():
        if a.GetAtomicNum() == 1:
            docked_copy.DeleteAtom(a)
    ## Calculate RMSD
    rmsd = oechem.OERMSD(lig, docked_copy)

    results = (
        lig_name,
        apo_name,
        f"{out_base}/docked.sdf",
        rmsd,
        posit_prob,
        chemgauss_score,
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

    ## Performance arguments
    parser.add_argument(
        "-n", default=10, type=int, help="Number of processors to use."
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
        compound_id_dict = get_compound_id_xtal_dicts(
            parse_fragalysis_data(args.x, frag_dir).values()
        )[1]

        ## Map all holo structure names to their ligand name
        all_holo_names = [
            f'{compound_id_dict[n.split("_")[0]]}_{n.split("_")[1]}'
            if n.split("_")[0] in compound_id_dict
            else n
            for n in all_holo_names
        ]

    print(f"{len(all_apo_fns)} apo structures")
    print(f"{len(all_holo_fns)} ligands to dock", flush=True)

    ## Get ligands from all holo structures
    all_ligs = [
        split_openeye_mol(load_openeye_pdb(fn))["lig"] for fn in all_holo_fns
    ]

    ## Get proteins from apo structures
    all_prots = [
        split_openeye_mol(load_openeye_pdb(fn))["pro"] for fn in all_apo_fns
    ]

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
        out_dir = f"{args.o}/{lig_name}/"
        os.makedirs(out_dir, exist_ok=True)
        ## Load and parse apo protein
        for prot_name, apo_prot in zip(all_apo_names, all_prots):
            mp_args.append(
                (apo_prot, lig, ref_prot, out_dir, prot_name, lig_name, args.du)
            )

    results_cols = [
        "SARS_ligand",
        "MERS_structure",
        "docked_file",
        "docked_RMSD",
        "POSIT_prob",
        "chemgauss4_score",
    ]
    nprocs = min(mp.cpu_count(), len(mp_args), args.n)
    print(f"Running {len(mp_args)} docking runs over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        results_df = pool.starmap(mp_func, mp_args)
    results_df = pandas.DataFrame(results_df, columns=results_cols)

    results_df.to_csv(f"{args.o}/all_results.csv")


if __name__ == "__main__":
    main()
