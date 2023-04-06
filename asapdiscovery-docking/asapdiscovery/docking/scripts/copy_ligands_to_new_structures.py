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
    dus = []
    for protein_file in protein_files:
        # Get protein name
        # TODO: replace this by fetching name directly from OEDU file
        protein_name = protein_file.stem

        if protein_file.suffix == ".oedu":
            # Load protein
            du = oechem.OEDesignUnit()
            if not oechem.OEReadDesignUnit(str(protein_file), du):
                logger.warning(f"Failed to read DesignUnit {protein_file}")
                continue
            du.SetTitle(protein_name)
            if not du.HasReceptor():
                logger.warning(
                    f"DesignUnit {protein_name} does not have a receptor; making one..."
                )
                oedocking.OEMakeReceptor(du)
            dus.append(du.CreateCopy())
        if protein_file.suffix == ".pdb":
            # Load protein
            mol = load_openeye_pdb(str(protein_file))
            if mol is None:
                logger.warning(f"Failed to read protein {protein_file}")
                continue
            mol.SetTitle(protein_name)
            # Make receptor
            du = oechem.OEDesignUnit()
            du.SetTitle(protein_name)
            oedocking.OEMakeReceptor(du, mol)
            dus.append(du)
    logger.info(f"Loaded {len(dus)} proteins from {args.protein_glob}")

    for mol in mols[0:1]:
        out_dir = output_dir / mol.GetTitle()
        if not out_dir.exists():
            out_dir.mkdir()

        # Use posit to dock against each DU
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
