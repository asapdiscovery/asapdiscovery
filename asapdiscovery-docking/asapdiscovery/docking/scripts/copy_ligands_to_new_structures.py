"""
This is the first step of the fauxalysis pipeline. It copies the ligands from the prepped structures 
to a new set of structures
"""
import argparse
from pathlib import Path
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    oechem,
    oespruce,
    oedocking,
    load_openeye_sdfs,
)
import multiprocessing as mp
from glob import glob


def get_args():
    parser = argparse.ArgumentParser(
        description="Copy ligands from prepped structures to new structures"
    )
    parser.add_argument(
        "-l", "--ligand_sdf", required=True, help="Path to ligand SDF file"
    )
    parser.add_argument(
        "-p", "--protein_glob", required=True, help="Glob string for protein pdb files"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-n", "--num_cores", default=1, type=int, help="Number of processes to use"
    )
    parser.add_argument(
        "--debug_num",
        type=int,
        default=-1,
        help="Number of tasks to pass to multiprocessing. Useful for debugging and testing.",
    )
    parser.add_argument(
        "--by_compound",
        action="store_true",
        default=False,
        help="If true, sort design units for each ligand, otherwise sort design units for each protein",
    )
    return parser.parse_args()


def make_dus_for_protein(prot_mol, lig_mols, output_dir):
    out_dir = output_dir / prot_mol.GetTitle()
    if not out_dir.exists():
        out_dir.mkdir()
    logger = FileLogger(
        f"copy_ligand_to_new_structures.{prot_mol.GetTitle()}", out_dir
    ).getLogger()

    for lig_mol in lig_mols:
        logger.info(f"Making DUs for {lig_mol.GetTitle()}")
        errfs = oechem.oeofstream(
            str(out_dir / f"openeye_{lig_mol.GetTitle()}_{prot_mol.GetTitle()}-log.txt")
        )
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)
        oechem.OEThrow.Info(f"Making DU for {lig_mol.GetTitle()}")

        du = oechem.OEDesignUnit()
        du.SetTitle(f"{prot_mol.GetTitle()}_{lig_mol.GetTitle()}")
        oespruce.OEMakeDesignUnit(du, prot_mol, lig_mol)
        logger.info(f"Making Receptor for {prot_mol.GetTitle()}")
        oedocking.OEMakeReceptor(du)
        out_fn = out_dir / f"{lig_mol.GetTitle()}_{prot_mol.GetTitle()}.oedu"
        oechem.OEWriteDesignUnit(str(out_fn), du)


def make_dus_for_ligand(lig_mol, prot_mols, output_dir):
    out_dir = output_dir / lig_mol.GetTitle()
    if not out_dir.exists():
        out_dir.mkdir()
    logger = FileLogger(
        f"copy_ligand_to_new_structures.{lig_mol.GetTitle()}", out_dir
    ).getLogger()

    for prot_mol in prot_mols:
        logger.info(f"Making DU for {prot_mol.GetTitle()}")
        errfs = oechem.oeofstream(
            str(out_dir / f"openeye_{prot_mol.GetTitle()}_{lig_mol.GetTitle()}-log.txt")
        )
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)
        oechem.OEThrow.Info(f"Making DU for {prot_mol.GetTitle()}")
        du = oechem.OEDesignUnit()
        du.SetTitle(f"{prot_mol.GetTitle()}_{lig_mol.GetTitle()}")
        oespruce.OEMakeDesignUnit(du, prot_mol, lig_mol)
        logger.info(f"Making Receptor for {prot_mol.GetTitle()}")
        oedocking.OEMakeReceptor(du)
        out_fn = out_dir / f"{lig_mol.GetTitle()}_{prot_mol.GetTitle()}.oedu"
        oechem.OEWriteDesignUnit(str(out_fn), du)


def main():
    # Make output directory
    args = get_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    logger = FileLogger(
        "copy_ligand_to_new_structures", path=str(output_dir)
    ).getLogger()

    # Load molecules
    mols = load_openeye_sdfs(args.ligand_sdf)
    logger.info(f"Loaded {len(mols)} ligands from {args.ligand_sdf}")

    # Load proteins
    protein_files = [Path(fn) for fn in glob(args.protein_glob)]
    prot_mols = []
    for protein_file in protein_files:
        # Get protein name
        # TODO: replace this by fetching name directly from OEDU file
        protein_name = protein_file.stem
        if protein_file.suffix == ".pdb":
            # Load protein
            mol = load_openeye_pdb(str(protein_file))
            if mol is None:
                logger.warning(f"Failed to read protein {protein_file}")
                continue
            mol.SetTitle(protein_name)
            prot_mols.append(mol)
        else:
            raise NotImplementedError("Only PDB files are supported for now")

    logger.info(f"Loaded {len(prot_mols)} proteins from {args.protein_glob}")

    if args.by_compound:
        mp_args = [(lig_mol, prot_mols, output_dir) for lig_mol in mols]
        func_ = make_dus_for_ligand
    else:
        mp_args = [(prot_mol, mols, output_dir) for prot_mol in prot_mols]
        func_ = make_dus_for_protein

    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    logger.info(f"CPUs: {mp.cpu_count()}")
    logger.info(f"N Processes: {len(mp_args)}")
    logger.info(f"N Cores: {args.num_cores}")

    mp_args = mp_args[: args.debug_num]
    logger.info(f"Running {len(mp_args)} tasks over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(func_, mp_args)

    logger.info("Done")


if __name__ == "__main__":
    main()
