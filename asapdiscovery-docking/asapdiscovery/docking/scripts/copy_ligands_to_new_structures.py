"""
This is the first step of the fauxalysis pipeline. It copies the ligands from the prepped structures 
to a new set of structures
"""
import argparse
from pathlib import Path
import logging
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    save_openeye_pdb,
    save_openeye_sdf,
    oechem,
    oedocking,
    oespruce,
    load_openeye_sdfs,
    combine_protein_ligand,
)
from asapdiscovery.docking.docking import run_docking_oe


def get_args():
    parser = argparse.ArgumentParser(
        description="Copy ligands from prepped structures to new structures"
    )
    parser.add_argument(
        "-l", "--ligand_sdf", required=True, help="Path to ligand SDF file"
    )
    parser.add_argument(
        "-p", "--protein_glob", required=True, help="Glob string for protein oedu files"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to output directory"
    )
    return parser.parse_args()


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
    protein_files = list(Path().glob(args.protein_glob))
    prot_mols = []
    for protein_file in protein_files[3:5]:
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

    for mol in mols[0:1]:
        out_dir = output_dir / mol.GetTitle()
        if not out_dir.exists():
            out_dir.mkdir()

        # Make new Receptors
        dus = []
        for prot_mol in prot_mols:
            logger.info(f"Making DU for {prot_mol.GetTitle()}")
            # combined = combine_protein_ligand(prot_mol, mol)
            du = oechem.OEDesignUnit()
            du.SetTitle(prot_mol.GetTitle())
            oespruce.OEMakeDesignUnit(du, prot_mol, mol)
            logger.info(f"Making Receptor for {prot_mol.GetTitle()}")
            oedocking.OEMakeReceptor(du)
            out_fn = out_dir / f"{mol.GetTitle()}_{prot_mol.GetTitle()}.oedu"
            oechem.OEWriteDesignUnit(str(out_fn), du)
            dus.append(du)
        # test reversed order
        dus.reverse()
        # Use posit to dock against each DU
        logger.info("Running docking for all DUs")
        success, posed_mol, docking_id = run_docking_oe(
            design_units=dus,
            orig_mol=mol,
            dock_sys="posit",
            relax="clash",
            hybrid=False,
            compound_name=mol.GetTitle(),
            use_omega=True,
            num_poses=1,
        )
        if success:
            out_fn = out_dir / "docked.sdf"
            save_openeye_sdf(posed_mol, str(out_fn))
    logger.info("Done")


if __name__ == "__main__":
    main()
