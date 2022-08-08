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

from kinoml.docking.OEDocking import pose_molecules

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from covid_moonshot_ml.datasets.utils import (
    load_openeye_pdb,
    save_openeye_pdb,
    save_openeye_sdf,
    split_openeye_mol,
)
from covid_moonshot_ml.docking.docking import (
    build_docking_system_direct,
    run_docking,
)
from covid_moonshot_ml.modeling import du_to_complex, make_du_from_new_lig


def mp_func(apo_prot, lig, ref_prot, out_dir, holo_name, save_du=False):
    out_base = f"{out_dir}/{holo_name}/"
    os.makedirs(out_base, exist_ok=True)
    out_fn = f"{out_base}/predocked"
    ## Make design unit and prep the receptor
    du = make_du_from_new_lig(apo_prot, lig, ref_prot, False, False)
    oedocking.OEMakeReceptor(du)

    ## Save if desired
    if save_du:
        oechem.OEWriteDesignUnit(f"{out_fn}.oedu", du)

    ## Get protein+lig complex in molecule form and save
    complex_mol = du_to_complex(du)
    save_openeye_pdb(complex_mol, f"{out_fn}.pdb")

    ## Run docking with kinoml
    dock_lig = oechem.OEGraphMol()
    du.GetLigand(dock_lig)
    docked_mol = pose_molecules(du, [dock_lig], score_pose=True)[0]

    save_openeye_sdf(docked_mol, f"{out_base}/docked.sdf")

    ## Calculate RMSD
    orig_mol = oechem.OEGraphMol()
    du.GetLigand(orig_mol)
    rmsd = oechem.OERMSD(orig_mol, docked_mol)

    return (
        name,
        holo_name,
        f"{out_base}/docked.sdf",
        rmsd,
        float(oechem.OEGetSDData(docked_mol, "POSIT::Probability")),
        float(oechem.OEGetSDData(docked_mol, "Chemgauss4")),
    )


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

    print(args.apo, args.holo)

    print(f"{len(all_apo_fns)} apo structures")
    print(f"{len(all_holo_fns)} ligands to dock", flush=True)

    ## Get ligands from all holo structures
    all_ligs = [
        split_openeye_mol(load_openeye_pdb(fn))["lig"] for fn in all_holo_fns
    ]

    ## Parse reference
    if args.ref:
        ref_prot = split_openeye_mol(load_openeye_pdb(args.ref))["pro"]
    else:
        ref_prot = None

    ## Figure out cache dir for docking
    if args.cache is None:
        cache_dir = f"{args.o}/.cache/"
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = args.cache


    mp_args = []
    ## Construct all args for mp_func
    for name, fn in zip(all_apo_names, all_apo_fns):
        out_dir = f"{args.o}/{name}/"
        os.makedirs(out_dir, exist_ok=True)
        ## Load and parse apo protein
        apo_prot = split_openeye_mol(load_openeye_pdb(fn))["pro"]
        for holo_name, lig in zip(all_holo_names, all_ligs):
            mp_args.append(
                (apo_prot, lig, ref_prot, out_dir, holo_name, args.du)
            )

            # reloaded_mol = load_openeye_pdb(f"{out_fn}.pdb")

            # ## Build docking system and run docking
            # docking_system = build_docking_system_direct(
            #     reloaded_mol,
            #     oechem.OEMolToSmiles(lig),
            #     f"{name}_{holo_name}",
            #     holo_name,
            # )

            # from kinoml.modeling.OEModeling import prepare_structure
            # from openeye import oespruce

            # print("complex_mol", flush=True)
            # print(
            #     # prepare_structure(
            #     #     complex_mol, has_ligand=True, loop_db=args.loop
            #     # )
            #     list(oespruce.OEMakeDesignUnits(complex_mol))
            # )

            # print("reloaded_mol", flush=True)
            # print(
            #     # prepare_structure(
            #     #     reloaded_mol, has_ligand=True, loop_db=args.loop
            #     # )
            #     list(oespruce.OEMakeDesignUnits(reloaded_mol))
            # )

            # print("docking_system", flush=True)
            # print(
            #     # prepare_structure(
            #     #     docking_system.protein.molecule,
            #     #     has_ligand=True,
            #     #     loop_db=args.loop,
            #     # )
            #     list(
            #         oespruce.OEMakeDesignUnits(docking_system.protein.molecule)
            #     )
            # )
            # return

            # ## Run docking
            # docking_system = run_docking(
            #     cache_dir, out_base, args.loop, 1, [docking_system]
            # )

            # print(
            #     "POSIT probability:",
            #     oechem.OEGetSDData(docked_mol, "POSIT::Probability"),
            #     flush=True
            # )
            # print(
            #     "Docking score:",
            #     oechem.OEGetSDData(docked_mol, "Chemgauss4"),
            #     flush=True
            # )

            # print(
            #     "Docking score:",
            #     docking_system.featurizations["last"]._topology.docking_score,
            #     flush=True,
            # )
            # print(
            #     "POSIT probability:",
            #     docking_system.featurizations[
            #         "last"
            #     ]._topology.posit_probability,
            #     flush=True,
            # )

    results_cols = [
        "MERS_structure",
        "SARS_structure",
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
