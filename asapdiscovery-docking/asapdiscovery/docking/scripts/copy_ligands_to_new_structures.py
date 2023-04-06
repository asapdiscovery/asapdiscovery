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
    dus = {}
    for protein_file in protein_files:
        # Get protein name
        # TODO: replace this by fetching name directly from OEDU file
        protein_name = protein_file.stem

        # Load protein
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(str(protein_file), du):
            logger.warning(f"Failed to read DesignUnit {protein_file}")
            continue
        du.SetTitle(protein_name)
        dus[protein_name] = du
    logger.info(f"Loaded {len(dus)} proteins from {args.protein_glob}")

    for mol in mols[0:1]:
        out_dir = output_dir / mol.GetTitle()

        # Use posit to dock against each DU
        for name, du in dus.items():
            success, posed_mol, docking_id = run_docking_oe(
                du=du,
                orig_mol=mol,
                dock_sys="posit",
                relax="clash",
                hybrid=True,
                compound_name=mol.GetTitle(),
                use_omega=True,
                num_poses=1,
            )
    #         if success:
    #             out_fn = os.path.join(out_dir, "docked.sdf")
    #             save_openeye_sdf(posed_mol, out_fn)
    #
    #             rmsds = []
    #             posit_probs = []
    #             posit_methods = []
    #             chemgauss_scores = []
    #
    #             for conf in posed_mol.GetConfs():
    #                 rmsds.append(float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_RMSD")))
    #                 posit_probs.append(
    #                     float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_POSIT"))
    #                 )
    #                 posit_methods.append(
    #                     oechem.OEGetSDData(conf, f"Docking_{docking_id}_POSIT_method")
    #                 )
    #                 chemgauss_scores.append(
    #                     float(oechem.OEGetSDData(conf, f"Docking_{docking_id}_Chemgauss4"))
    #                 )
    #             smiles = oechem.OEGetSDData(conf, "SMILES")
    #             clash = int(oechem.OEGetSDData(conf, f"Docking_{docking_id}_clash"))
    #         else:
    #             out_fn = ""
    #             rmsds = [-1.0]
    #             posit_probs = [-1.0]
    #             posit_methods = [""]
    #             chemgauss_scores = [-1.0]
    #             clash = -1
    #             smiles = "None"
    #
    #         results = [
    #             (
    #                 lig_name,
    #                 du_name,
    #                 out_fn,
    #                 i,
    #                 rmsd,
    #                 prob,
    #                 method,
    #                 chemgauss,
    #                 clash,
    #                 smiles,
    #             )
    #             for i, (rmsd, prob, method, chemgauss) in enumerate(
    #                 zip(rmsds, posit_probs, posit_methods, chemgauss_scores)
    #             )
    #         ]
    #
    #         pkl.dump(results, open(os.path.join(out_dir, "results.pkl"), "wb"))
    # # Combine protein and ligand
    # combined = combine_protein_ligand(prot, mol)
    # # Save combined molecule
    # save_openeye_pdb(combined, str(output_dir / f"{mol.GetTitle()}.pdb"))
    # logger.info(f"Saved {mol.GetTitle()}.pdb")
    logger.info("Done")


if __name__ == "__main__":
    main()
