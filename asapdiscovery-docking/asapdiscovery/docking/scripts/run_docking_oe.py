"""
Script to dock an SDF file of ligands to prepared structures. Example usage:
python run_docking_oe.py \
-e /path/to/experimental_data.json \
-r /path/to/receptors/*.oedu \
-s /path/to/mcs_sort_results.pkl \
-o /path/to/docking/output/

Example usage with a custom regex for parsing your DU filenames:
Suppose your receptors are named
 - /path/to/receptors/Structure0_0.oedu
 - /path/to/receptors/Structure0_1.oedu
 - /path/to/receptors/Structure0_2.oedu
 - ...
where each Structure<i> is a unique crystal structure, and each Structure<i>_<j> is a
different DesignUnit for that structure. You might construct your regex as
'(Compound[0-9]+)_[0-9]+', which will capture the Structure<i> as the unique structure
ID, and Structure<i>_<j> as the full name. Note that single quotes should be used around
the regex in order to avoid any accidental wildcard expansion by the OS:
python run_docking_oe.py \
-e /path/to/experimental_data.json \
-r /path/to/receptors/*.oedu \
-s /path/to/mcs_sort_results.pkl \
-o /path/to/docking/output/ \
-re '(Compound[0-9]+)_[0-9]+'
"""
import argparse
import multiprocessing as mp
import os
import pickle as pkl
import shutil
from glob import glob

import pandas
from asapdiscovery.data.openeye import load_openeye_sdf  # noqa: E402
from asapdiscovery.data.openeye import save_openeye_sdf  # noqa: E402
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.utils import check_filelist_has_elements  # noqa: E402
from asapdiscovery.docking.docking import run_docking_oe  # noqa: E402


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


def load_dus(fn_dict):
    """
    Load all present oedu files.

    Parameters
    ----------
    fn_dict : Dict[str, str]
        Dictionary mapping full DesignUnit name (with chain) to full filename

    Returns
    -------
    Dict[str, oechem.OEDesignUnit]
        Dictionary mapping full Mpro name/compound id (including chain) to its
        design unit
    """

    du_dict = {}
    for full_name, fn in fn_dict.items():
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(fn, du):
            print(f"Failed to read DesignUnit {fn}", flush=True)
            continue
        du_dict[full_name] = du

    print(f"{len(du_dict.keys())} design units loaded")
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
        print(f"Loading found results for {lig_name}_{du_name}", flush=True)
        return pkl.load(open(os.path.join(out_dir, "results.pkl"), "rb"))
    os.makedirs(out_dir, exist_ok=True)

    success, posed_mol, docking_id = run_docking_oe(*args, **kwargs)
    if success:
        out_fn = os.path.join(out_dir, "docked.sdf")
        save_openeye_sdf(posed_mol, out_fn)

        rmsds = []
        posit_probs = []
        posit_methods = []
        chemgauss_scores = []

        for conf in posed_mol.GetConfs():
            rmsds.append(float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_RMSD")))
            posit_probs.append(
                float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_POSIT"))
            )
            posit_methods.append(
                oechem.OEGetSDData(conf, f"Docking_{docking_id}_POSIT_method")
            )
            chemgauss_scores.append(
                float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_Chemgauss4"))
            )
        smiles = oechem.OEGetSDData(conf, "SMILES")
        clash = int(oechem.OEGetSDData(conf, f"Docking_{docking_id}_clash"))
    else:
        out_fn = ""
        rmsds = [-1.0]
        posit_probs = [-1.0]
        posit_methods = [""]
        chemgauss_scores = [-1.0]
        clash = -1
        smiles = "None"

    results = [
        (
            lig_name,
            du_name,
            out_fn,
            i,
            rmsd,
            prob,
            method,
            chemgauss,
            clash,
            smiles,
        )
        for i, (rmsd, prob, method, chemgauss) in enumerate(
            zip(rmsds, posit_probs, posit_methods, chemgauss_scores)
        )
    ]

    pkl.dump(results, open(os.path.join(out_dir, "results.pkl"), "wb"))
    return results


def parse_du_filenames(receptors, regex, basefile="predocked.oedu"):
    """
    Parse list of DesignUnit filenames and extract identifiers using the given regex.
    `regex` should have one capturing group (which can be the entire string if desired).

    Parameters
    ----------
    receptors : Union[List[str], str]
        Either list of DesignUnit filenames, or glob/directory/file to load from. If a
        file is passed, will assume this is a CSV file and will load from the
        "Docked_File" column using `basefile`
    regex : str
        Regex string for parsing
    basefile : str, default="predocked.oedu"
        If a CSV file is passed for `receptors`, this is the base filename that will be
        appended to every directory found in the "Docked_File" column

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping Mpro dataset name/compound id to list of full
        Mpro names/compound ids (with chain)
    Dict[str, str]
        Dictionary mapping full name (with chain) to full filename
    """
    from asapdiscovery.data.utils import construct_regex_function

    # First get full list of filenames
    if type(receptors) is list:
        print("Using files as given")
        all_fns = receptors
    elif os.path.isdir(receptors):
        print(f"Using {receptors} as directory")
        all_fns = [
            os.path.join(receptors, fn)
            for _, _, files in os.walk(receptors)
            for fn in files
            if fn[-4:] == "oedu"
        ]
    elif os.path.isfile(receptors):
        print(f"Using {receptors} as file")
        df = pandas.read_csv(receptors)
        try:
            all_fns = [
                os.path.join(os.path.dirname(fn), basefile) for fn in df["Docked_File"]
            ]
        except KeyError:
            raise ValueError("Docked_File column not found in given CSV file.")
    else:
        print(f"Using {receptors} as glob")
        all_fns = glob(receptors)

    # check that we actually have loaded in prepped receptors.
    check_filelist_has_elements(all_fns, tag="prepped receptors")
    print(f"{len(all_fns)} DesignUnit files found", flush=True)

    # Build regex search function
    regex_func = construct_regex_function(regex, ret_groups=True)
    # Perform searches and build dicts
    dataset_dict = {}
    fn_dict = {}
    for fn in all_fns:
        try:
            full_name, dataset = regex_func(fn)
        except ValueError:
            print(f"No regex match found for {fn}", flush=True)
            continue

        try:
            dataset = dataset[0]
        except IndexError:
            raise ValueError(f"No capturing group in regex {regex}")

        try:
            dataset_dict[dataset].append(full_name)
        except KeyError:
            dataset_dict[dataset] = [full_name]
        fn_dict[full_name] = fn

    return dataset_dict, fn_dict


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
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
        nargs="+",
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
    parser.add_argument(
        "-re",
        "--regex",
        help=(
            "Regex for extracting DesignUnit identifiers from the "
            "OpenEye DesignUnit filenames."
        ),
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    # Performance arguments
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
    parser.add_argument(
        "-g",
        "--omega",
        action="store_true",
        help="Use Omega conformer enumeration.",
    )
    parser.add_argument(
        "-p",
        "--num_poses",
        type=int,
        default=1,
        help="Number of poses to return from docking.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Parse symlinks in output_dir
    args.output_dir = os.path.realpath(args.output_dir)

    if args.exp_file:
        import json

        from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate

        # Load compounds
        exp_compounds = [
            c
            for c in ExperimentalCompoundDataUpdate(
                **json.load(open(args.exp_file))
            ).compounds
            if c.smiles is not None
        ]
        # Make OEGraphMol for each compound
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
            # Load all ligands to dock
            ifs = oechem.oemolistream()
            ifs.open(args.lig_file)
            mols = [mol.CreateCopy() for mol in ifs.GetOEGraphMols()]
    elif args.exp_file is None:
        raise ValueError("Need to specify exactly one of --exp_file or --lig_file.")
    n_mols = len(mols)

    # The receptor args are captured as a list, but we still want to handle the case of
    #  a glob/directory/filename being passed. If there's only one thing in the list,
    #  assume it is a glob/directory/filename, and pull it out of the list so it's
    #  properly handled in `parse_du_filenames`
    if len(args.receptor) == 1:
        args.receptor = args.receptor[0]
    # Handle default regex
    if args.regex is None:
        if args.by_compound:
            from asapdiscovery.data.utils import MOONSHOT_CDD_ID_REGEX_CAPT

            args.regex = MOONSHOT_CDD_ID_REGEX_CAPT
        else:
            from asapdiscovery.data.utils import MPRO_ID_REGEX_CAPT

            args.regex = MPRO_ID_REGEX_CAPT
    dataset_dict, fn_dict = parse_du_filenames(args.receptor, args.regex)

    # Load all receptor DesignUnits
    du_dict = load_dus(fn_dict)
    print(f"{n_mols} molecules found")
    print(f"{len(du_dict.keys())} receptor structures found")
    assert n_mols > 0
    assert len(du_dict.keys()) > 0

    # Load sort indices if given
    if args.sort_res:
        compound_ids, xtal_ids, sort_idxs = pkl.load(open(args.sort_res, "rb"))
        # If we're docking to all DUs, set top_n appropriately
        if args.top_n == -1:
            args.top_n = len(xtal_ids)

        # Make sure that compound_ids match with experimental data if that's
        #  what we're using
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
        # Check to see if the SDF files have a Compound_ID Column
        if all(len(oechem.OEGetSDData(mol, "Compound_ID")) > 0 for mol in mols):
            print("Using Compound_ID column from sdf file")
            compound_ids = [oechem.OEGetSDData(mol, "Compound_ID") for mol in mols]
        else:
            # Use index as compound_id
            compound_ids = [str(i) for i in range(n_mols)]
        # Get dataset values from DesignUnit filenames
        xtal_ids = list(dataset_dict.keys())
        # Arbitrary sort index, same for each ligand
        sort_idxs = [list(range(len(xtal_ids)))] * n_mols
        args.top_n = len(xtal_ids)
    mp_args = []
    for i, m in enumerate(mols):
        dock_dus = []
        xtals = []
        for xtal in sort_idxs[i][: args.top_n]:
            if xtal_ids[xtal] not in dataset_dict:
                continue
            # Get the DU for each full Mpro name associated with this dataset
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
                args.omega,
                args.num_poses,
            )
            for du, x in zip(dock_dus, xtals)
        ]
        mp_args.extend(new_args)

    results_cols = [
        "ligand_id",
        "du_structure",
        "docked_file",
        "pose_id",
        "docked_RMSD",
        "POSIT_prob",
        "POSIT_method",
        "chemgauss4_score",
        "clash",
        "SMILES",
    ]
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(
        f"CPUs: {mp.cpu_count()}\n"
        f"N Processes: {mp_args}\n"
        f"N Cores: {args.num_cores}"
    )
    print(f"Running {len(mp_args)} docking runs over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        results_df = pool.starmap(mp_func, mp_args)
    results_df = [res for res_list in results_df for res in res_list]
    results_df = pandas.DataFrame(results_df, columns=results_cols)

    results_df.to_csv(f"{args.output_dir}/all_results.csv")

    # Concatenate all individual SDF files
    combined_sdf = f"{args.output_dir}/combined.sdf"
    with open(combined_sdf, "wb") as wfd:
        for f in results_df["docked_file"]:
            if f == "":
                continue
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)


if __name__ == "__main__":
    main()
