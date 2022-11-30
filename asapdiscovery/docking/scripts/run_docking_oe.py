"""
Script to dock an SDF file of ligands to prepared structures.
Example Usage:
    python ~/covid-moonshot-ml/asapdiscovery/docking/scripts/run_docking_oe.py \
        -l ~/asap-datasets/test_run_oe/ligand.sdf \
        -r '/lila/data/chodera/asap-datasets/full_frag_prepped_dus_seqres/*/prepped_receptor.oedu' \
        -s /lila/data/chodera/asap-datasets/amines_small_mcs/mcs_sort_index.pkl \
        -o ~/asap-datasets/test_run_oe/ \
        -n 10 \
        -t 10

Example Cluster Scripts (in cluster_scripts):
    test_run_oe_docking.sh
    run_oe_docking_amines_small.sh
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

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
print(sys.path)
from asapdiscovery.data.utils import load_openeye_sdf, save_openeye_sdf
from asapdiscovery.docking.docking import run_docking_oe
from asapdiscovery.docking.analysis import DockingResults


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


def load_dus(file_base, by_compound=False):
    """
    Load all present oedu files. If `file_base` is a directory, os.walk will be
    used to find all .oedu files in the directory. Otherwise, it will be
    assessed with glob.

    Parameters
    ----------
    file_base : str
        Directory/base filepath for .oedu files, or best_results.csv file if
        `by_compound` is True.
    by_compound : bool, default=False
        Whether to load by dataset (False) or by compound_id (True).

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping Mpro dataset name/compound id to list of full
        Mpro names/compound ids (with chain)
    Dict[str, oechem.OEDesignUnit]
        Dictionary mapping full Mpro name/compound id (including chain) to its
        design unit
    """
    print(f"Loading receptor files")
    if os.path.isdir(file_base):
        print(f"\tReading {file_base} as base path")
        all_fns = [
            os.path.join(file_base, fn)
            for _, _, files in os.walk(file_base)
            for fn in files
            if fn[-4:] == "oedu"
        ]
    elif os.path.isfile(file_base) and by_compound:
        print(f"\tReading {file_base} as best_results.csv")
        df = pandas.read_csv(file_base)
        all_fns = [
            os.path.join(os.path.dirname(fn), "predocked.oedu")
            for fn in df["Docked_File"]
        ]
    else:
        print(f"\tUsing glob to parse {file_base}")
        all_fns = glob(file_base)

    assert (
        len(all_fns) > 0
    ), f"ERROR: The list of files parsed from {file_base} is empty."

    du_dict = {}
    dataset_dict = {}
    if by_compound:
        re_pat = r"([A-Z]{3}-[A-Z]{3}-[a-z0-9]+-[0-9]+)_[0-9][A-Z]"
    else:
        re_pat = r"(Mpro-[A-Za-z][0-9]+)_[0-9][A-Z]"
    for fn in all_fns:
        m = re.search(re_pat, fn)
        if m is None:
            search_type = "compound_id" if by_compound else "Mpro dataset"
            print(f"No {search_type} found for {fn}", flush=True)
            continue

        dataset = m.groups()[0]
        full_name = m.group()
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(fn, du):
            print(f"Failed to read DesignUnit {fn}", flush=True)
            continue
        du_dict[full_name] = du
        try:
            dataset_dict[dataset].append(full_name)
        except KeyError:
            dataset_dict[dataset] = [full_name]
    assert len(dataset_dict) > 0, f"Dataset dictionary is empty!"
    assert len(du_dict) > 0, f"Design Unit dictionary is empty!"
    return dataset_dict, du_dict


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
        print(f"Loading found results for {lig_name}_{du_name}", flush=True)
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
        smiles = oechem.OEGetSDData(posed_mol, f"SMILES")
    else:
        out_fn = ""
        rmsd = -1.0
        posit_prob = -1.0
        chemgauss_score = -1.0
        clash = -1
        smiles = "None"

    results = (
        lig_name,
        du_name,
        out_fn,
        rmsd,
        posit_prob,
        chemgauss_score,
        clash,
        smiles,
    )
    pkl.dump(results, open(os.path.join(out_dir, "results.pkl"), "wb"))
    return results


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument("-l", "--lig_file", help="SDF file containing ligands.")
    parser.add_argument(
        "-e",
        "--exp_file",
        help="JSON file containing ExperimentalCompoundDataUpdate object.",
    )
    parser.add_argument(
        "-r",
        "--receptor",
        required=True,
        help=(
            "Path/glob to prepped receptor(s), or best_results.csv file if "
            "--by_compound is given."
        ),
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
    parser.add_argument(
        "-c",
        "--by_compound",
        action="store_true",
        help="Load/store DesignUnits by compound_id instead of by Mpro dataset.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    ## Parse symlinks in output_dir
    args.output_dir = os.path.realpath(args.output_dir)

    if args.exp_file:
        import json
        from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate

        ## Load compounds
        exp_compounds = [
            c
            for c in ExperimentalCompoundDataUpdate(
                **json.load(open(args.exp_file, "r"))
            ).compounds
            if c.smiles is not None
        ]
        ## Make OEGraphMol for each compound
        mols = []
        for c in exp_compounds:
            new_mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(new_mol, c.smiles)
            mols.append(new_mol)
    if args.lig_file:
        if args.exp_file:
            print(
                (
                    "WARNING: Arguments passed for both --exp_file and "
                    "--lig_file, using --exp_file."
                ),
                flush=True,
            )
        else:
            ## Load all ligands to dock
            ifs = oechem.oemolistream()
            ifs.open(args.lig_file)
            mols = [mol.CreateCopy() for mol in ifs.GetOEGraphMols()]
    elif args.exp_file is None:
        raise ValueError(
            "Need to specify exactly one of --exp_file or --lig_file."
        )
    n_mols = len(mols)

    ## Load all receptor DesignUnits
    dataset_dict, du_dict = load_dus(args.receptor, args.by_compound)

    ## Load sort indices if given
    if args.sort_res:
        compound_ids, xtal_ids, sort_idxs = pkl.load(open(args.sort_res, "rb"))
        ## If we're docking to all DUs, set top_n appropriately
        if args.top_n == -1:
            args.top_n = len(xtal_ids)

        ## Make sure that compound_ids match with experimental data if that's
        ##  what we're using
        if args.exp_file:
            assert all(
                [
                    compound_id == c.compound_id
                    for (compound_id, c) in zip(compound_ids, exp_compounds)
                ]
            ), (
                "Sort result compound_ids are not equivalent to "
                "compound_ids in --exp_file."
            )
    else:
        ## Use index as compound_id
        compound_ids = [str(i) for i in range(n_mols)]
        ## Get dataset values from DesignUnit filenames
        xtal_ids = list(dataset_dict.keys())
        ## Arbitrary sort index, same for each ligand
        sort_idxs = [list(range(len(xtal_ids)))] * n_mols
        args.top_n = len(xtal_ids)

    mp_args = []
    for i, m in enumerate(mols):
        dock_dus = []
        xtals = []
        for xtal in sort_idxs[i][: args.top_n]:
            if xtal_ids[xtal] not in dataset_dict:
                print(f"{xtal_ids[xtal]} not found in dataset dictionary")
                continue
            ## Get the DU for each full Mpro name associated with this dataset
            dock_dus.extend([du_dict[x] for x in dataset_dict[xtal_ids[xtal]]])
            xtals.extend(dataset_dict[xtal_ids[xtal]])
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
        "SMILES",
    ]
    print(
        f"CPU Count:        {mp.cpu_count()} "
        f"Processes to Run: {len(mp_args)}"
        f"Number of Cores:  {args.num_cores}"
    )
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
