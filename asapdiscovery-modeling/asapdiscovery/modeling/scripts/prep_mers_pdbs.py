"""
This script prepares MERS biological assembly cif1 files downloaded from the PDB and prepared spruced design units.

Loading CIF1 Files:
    Currently it handles this by loading with OpenMM PDBxFile object and saving to a pdb file.
    CIF1 files are used because biological assembly files in pdb format come with two states,
    which OpenEye doesn't handle well.
    However, currently there is a bug in OpenEye that causes it to fail due to some alternate locations in the files.
    The OpenMM PDBxFile object seems to handle these without problem.

Protein preparation:
    The protein is 1) aligned, 2) mutated to the canonical mers sequence, and 3) prepared using spruce using OpenEye
    toolkit methods.

Example usage is found in ..cluster_scripts/prep_mers_pdbs.sh

"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import yaml
from asapdiscovery.data.openeye import load_openeye_cif1, oechem, save_openeye_pdb
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.modeling.modeling import (
    align_receptor,
    mutate_residues,
    spruce_protein,
)


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to downloaded cif1 files.",
    )
    parser.add_argument(
        "-r",
        "--ref_prot",
        default="../tests/prep_mers_rcsb/inputs/reference.pdb",
        type=str,
        help="Path to reference pdb to align to. If None, no alignment will be performed",
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    # Model-building arguments
    parser.add_argument(
        "-l",
        "--loop_db",
        default="/Users/alexpayne/Scientific_Projects/mers-drug-discovery/spruce_bace.loop_db",
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        default="../../../../metadata/mpro_mers_seqres.yaml",
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "--protein_only",
        action="store_true",
        default=True,
        help="If true, generate design units with only the protein in them",
    )
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )
    return parser.parse_args()


def prep_mp(cifpath, output, loop_db, ref_prot, seqres_yaml):
    # Set up logger
    name = str(cifpath.stem)
    logfile = output / f"{name}-log.txt"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(str(logfile), mode="w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set up logging for OE Functions to redirect Info / Warnings / etc from stdout
    errfs = oechem.oeofstream(str(output / f"{name}-oelog.txt"))
    oechem.OEThrow.SetOutputStream(errfs)

    du_fn = output / f"{name}-prepped_receptor_0.oedu"
    if du_fn.exists():
        logger.info(f"Already made {du_fn}!")
        return
    else:
        logger.info(f"Preparing {cifpath}")

    logger.info("Loading cif, saving to PDB with OpenMM, and loading with OpenEye")
    prot = load_openeye_cif1(str(cifpath))

    logger.info("Aligning to ref")

    ref_path = Path(ref_prot)
    prot = align_receptor(
        initial_complex=prot,
        ref_prot=ref_path.as_posix(),
        dimer=True,
        split_initial_complex=True,
        mobile_chain="A",  # TODO: make this not hardcoded? not sure what logic to use though
        ref_chain="A",
    )
    # aligned = str(output / f"{name}-01.pdb")
    # save_openeye_pdb(prot, aligned)

    logger.info("Preparing Sprucing options")
    loop_path = Path(loop_db)

    seqres_path = Path(seqres_yaml)
    with open(seqres_path) as f:
        seqres_dict = yaml.safe_load(f)
    seqres = seqres_dict["SEQRES"]

    res_list = seqres_to_res_list(seqres)
    seqres = " ".join(res_list)

    logger.info("Making mutations")

    prot = mutate_residues(prot, res_list, place_h=True)

    logger.info("Sprucing protein")

    du = spruce_protein(
        initial_prot=prot,
        seqres=seqres,
        loop_db=loop_path,
        return_du=True,
    )

    if type(du) == oechem.OEDesignUnit:
        logger.info("Saving Design Unit")

        du_fn = output / f"{name}-prepped_receptor_0.oedu"
        oechem.OEWriteDesignUnit(str(du_fn), du)

        logger.info("Saving PDB")

        prot = oechem.OEGraphMol()
        du.GetProtein(prot)

        # Add SEQRES entries if they're not present
        if (not oechem.OEHasPDBData(prot, "SEQRES")) and seqres:
            for seqres_line in seqres.split("\n"):
                if seqres_line != "":
                    oechem.OEAddPDBData(prot, "SEQRES", seqres_line[6:])

        prot_fn = output / f"{name}-prepped_receptor_0.pdb"
        save_openeye_pdb(prot, str(prot_fn))

    elif type(du) == oechem.OEGraphMol:
        logger.info("Design Unit preparation failed. Saving spruced protein")
        prot_fn = output / f"{name}-failed-spruced.pdb"
        save_openeye_pdb(du, str(prot_fn))


def main():
    args = get_args()

    inputs = Path(args.structure_dir)
    cifpaths = inputs.glob("*-assembly1.cif")
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True)

    mp_args = [
        [cifpath, output, args.loop_db, args.ref_prot, args.seqres_yaml]
        for cifpath in cifpaths
    ]
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
