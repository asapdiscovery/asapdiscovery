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
    oechem,
    load_openeye_sdfs,
    combine_protein_ligand,
)


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
    logger.info(f"Loaded {len(protein_files)} proteins from {args.protein_glob}")

    for protein_file in protein_files:
        # Load protein
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(str(protein_file), du):
            logger.warning(f"Failed to read DesignUnit {protein_file}")
            continue
        prot = oechem.OEGraphMol()
        du.GetProtein(prot)

        for mol in mols:
            # Combine protein and ligand
            combined = combine_protein_ligand(prot, mol)
            # Save combined molecule
            save_openeye_pdb(combined, str(output_dir / f"{mol.GetTitle()}.pdb"))
            logger.info(f"Saved {mol.GetTitle()}.pdb")


if __name__ == "__main__":
    main()
