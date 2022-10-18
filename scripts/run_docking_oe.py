"""
Script to dock an SDF file of ligands to prepared structures.
"""
import argparse
from glob import glob
import multiprocessing as mp
from openeye import oechem
import os
import pandas
import pickle as pkl
import re
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from covid_moonshot_ml.datasets.utils import load_openeye_sdf, save_openeye_sdf
from covid_moonshot_ml.docking.docking import run_docking_oe


def check_results(d):
    """
    Check if results exist already so we can skip.

    Parameters
    ----------
    d : str
        Directory

    Returns
    -------
    bool
        Results already exist
    """
    if (not os.path.isfile(os.path.join(d, "docked.sdf"))) or (
        not os.path.isfile(os.path.join(d, "results.pkl"))
    ):
        return False

    try:
        _ = load_openeye_sdf(os.path.join(d, "docked.sdf"))
    except Exception:
        return False

    try:
        _ = pkl.load(open(os.path.join(d, "results.pkl"), "rb"))
    except Exception:
        return False

    return True


def load_dus(file_base):
    """
    Load all present oedu files. If `file_base` is a directory, os.walk will be
    used to find all .oedu files in the directory. Otherwise, it will be
    assessed with glob.

    Parameters
    ----------
    file_base : str
        Directory/base filepath for .oedu files.

    Returns
    -------
    Dict[str, List[oechem.OEDesignUnit]]
        Dictionary mapping dataset to list of design units
    """

    if os.path.isdir(file_base):
        all_fns = [
            os.path.join(file_base, fn)
            for _, _, files in os.walk(file_base)
            for fn in files
            if fn[-4:] == "oedu"
        ]
    else:
        all_fns = glob(file_base)

    du_dict = {}
    re_pat = r"(Mpro-[A-Za-z][0-9]+)_"
    for fn in all_fns:
        dataset = re.search(re_pat, fn)
        if dataset is None:
            print(f"No Mpro dataset found for {fn}", flush=True)
            continue

        dataset = dataset.groups()[0]
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(fn, du):
            print(f"Failed to read DesignUnit {fn}", flush=True)
            continue
        try:
            du_dict[dataset].append(du)
        except KeyError:
            du_dict[dataset] = [du]

    return du_dict


def mp_func(out_dir, lig_name, du_name, *args, **kwargs):
    """
    Wrapper function for multiprocessing. Everything other than the named args
    will be passed directly to run_docking_oe.

    Parameters
    ----------
    out_dir : str
        Output file
    lig_name : str
        Ligand name
    du_name : str
        DesignUnit name

    Returns
    -------
    """
    if check_results(out_dir):
        return pkl.load(open(os.path.join(out_dir, "results.pkl"), "rb"))
    os.makedirs(out_dir, exist_ok=True)

    success, posed_mol, docking_id = run_docking_oe(*args, **kwargs)
    if success:
        out_fn = os.path.join(out_dir, "docked.sdf")
        save_openeye_sdf(posed_mol, out_fn)
        # with open(out_fn, "wb") as fp:
        #     fp.write(oechem.OEWriteMolToBytes(".sdf", posed_mol))

        rmsd = float(
            oechem.OEGetSDData(posed_mol, f"Docking_{docking_id}_RMSD")
        )
        posit_prob = float(
            oechem.OEGetSDData(posed_mol, f"Docking_{docking_id}_POSIT")
        )
        chemgauss_score = float(
            oechem.OEGetSDData(posed_mol, f"Docking_{docking_id}_Chemgauss4")
        )
        clash = int(
            oechem.OEGetSDData(posed_mol, f"Docking_{docking_id}_clash")
        )
    else:
        out_fn = ""
        rmsd = -1.0
        posit_prob = -1.0
        chemgauss_score = -1.0
        clash = -1

    results = (
        lig_name,
        du_name,
        out_fn,
        rmsd,
        posit_prob,
        chemgauss_score,
        clash,
    )
    pkl.dump(results, open(os.path.join(out_dir, "results.pkl"), "wb"))
    return results


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-l", "--lig_file", required=True, help="SDF file containing ligands."
    )
    parser.add_argument(
        "-r",
        "--receptor",
        required=True,
        help="Path/glob to prepped receptor(s).",
    )
    parser.add_argument(
        "-s",
        "--sort_res",
        help="Pickle file giving compound_ids, xtal_ids, and sort_idxs.",
    )

    ## Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    ## Performance arguments
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )
    parser.add_argument(
        "-t",
        "--top_n",
        type=int,
        default=1,
        help="Number of top matches to dock. Set to -1 to dock all.",
    )
    parser.add_argument(
        "-d",
        "--docking_sys",
        default="posit",
        help="Which docking system to use [posit, hybrid]. Defaults to posit.",
    )
    parser.add_argument(
        "-x",
        "--relax",
        default="none",
        help="When to run relaxation [none, clash, all]. Defaults to none.",
    )
    parser.add_argument(
        "-y",
        "--hybrid",
        action="store_true",
        help="Whether to only use hybrid docking protocol in POSIT.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## Load all ligands to dock
    ifs = oechem.oemolistream()
    ifs.open(args.lig_file)
    mols = [mol.CreateCopy() for mol in ifs.GetOEGraphMols()]
    n_mols = len(mols)

    ## Load all receptor DesignUnits
    all_dus = load_dus(args.receptor)

    ## Load sort indices if given
    if args.sort_res:
        compound_ids, xtal_ids, sort_idxs = pkl.load(open(args.sort_res, "rb"))
        ## If we're docking to all DUs, set top_n appropriately
        if args.top_n == -1:
            args.top_n = len(xtal_ids)
    else:
        ## Use index as compound_id
        compound_ids = [str(i) for i in range(n_mols)]
        ## Get dataset values from DesignUnit filenames
        xtal_ids = list(all_dus.keys())
        ## Arbitrary sort index, same for each ligand
        sort_idxs = [list(range(len(xtal_ids)))] * n_mols
        args.top_n = len(xtal_ids)

    mp_args = []
    for i, m in enumerate(mols):
        dock_dus = []
        xtals = []
        for xtal in sort_idxs[i][: args.top_n]:
            if xtal_ids[xtal] not in all_dus:
                continue
            dock_dus.extend(all_dus[xtal_ids[xtal]])
            xtals.extend(
                [
                    f"{xtal_ids[xtal]}_{j}"
                    for j in range(len(all_dus[xtal_ids[xtal]]))
                ]
            )
        new_args = [
            (
                os.path.join(args.output_dir, f"{compound_ids[i]}_{x}"),
                compound_ids[i],
                x,
                du,
                m,
                args.docking_sys.lower(),
                args.relax.lower(),
                args.hybrid,
                f"{compound_ids[i]}_{x}",
            )
            for du, x in zip(dock_dus, xtals)
        ]
        mp_args.extend(new_args)

    results_cols = [
        "ligand_id",
        "du_structure",
        "docked_file",
        "docked_RMSD",
        "POSIT_prob",
        "chemgauss4_score",
        "clash",
    ]
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Running {len(mp_args)} docking runs over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        results_df = pool.starmap(mp_func, mp_args)
    results_df = pandas.DataFrame(results_df, columns=results_cols)

    results_df.to_csv(f"{args.output_dir}/all_results.csv")

    ## Concatenate all individual SDF files
    combined_sdf = f"{args.output_dir}/combined.sdf"
    with open(combined_sdf, "wb") as wfd:
        for f in results_df["docked_file"]:
            if f == "":
                continue
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)


if __name__ == "__main__":
    main()
