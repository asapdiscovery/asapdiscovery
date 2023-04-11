from pathlib import Path
import pytest
from asapdiscovery.docking import run_docking_oe
from asapdiscovery.data.openeye import (
    load_openeye_sdfs,
    load_openeye_pdb,
    save_openeye_pdb,
    save_openeye_sdf,
    oechem,
    oedocking,
    oespruce,
    combine_protein_ligand,
)
from asapdiscovery.data.logging import FileLogger


def test_multi_receptor_docking():
    # Make output directory
    output_dir = Path("outputs/fauxalysis_generation_test")
    if not output_dir.exists():
        output_dir.mkdir()

    # Load molecules
    ligand_sdf = "inputs/Mpro_combined_labeled.sdf"
    mols = load_openeye_sdfs(ligand_sdf)
    print(f"Loaded {len(mols)} ligands from {ligand_sdf}")

    # Load proteins
    protein_glob = "inputs/prepped_mers_receptors/*.pdb"
    protein_files = list(Path().glob(protein_glob))
    prot_mols = []
    for protein_file in protein_files[3:5]:
        # Get protein name
        # TODO: replace this by fetching name directly from OEDU file
        protein_name = protein_file.stem
        if protein_file.suffix == ".pdb":
            # Load protein
            mol = load_openeye_pdb(str(protein_file))
            if mol is None:
                raise RuntimeError(f"Failed to read protein {protein_file}")
            mol.SetTitle(protein_name)
            prot_mols.append(mol)
        else:
            raise NotImplementedError("Only PDB files are supported for now")

    print(f"Loaded {len(prot_mols)} proteins from {protein_glob}")

    for mol in mols[0:1]:
        out_dir = output_dir / mol.GetTitle()
        if not out_dir.exists():
            out_dir.mkdir()

        # Make new Receptors
        dus = []
        for prot_mol in prot_mols:
            print(f"Making DU for {prot_mol.GetTitle()}")
            # combined = combine_protein_ligand(prot_mol, mol)
            du = oechem.OEDesignUnit()
            du.SetTitle(prot_mol.GetTitle())
            oespruce.OEMakeDesignUnit(du, prot_mol, mol)
            print(f"Making Receptor for {prot_mol.GetTitle()}")
            oedocking.OEMakeReceptor(du)
            out_fn = out_dir / f"{mol.GetTitle()}_{prot_mol.GetTitle()}.oedu"
            oechem.OEWriteDesignUnit(str(out_fn), du)
            dus.append(du)

        # Use posit to dock against each DU
        print("Running docking for all DUs")
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
        out_fn = out_dir / "docked.sdf"
        save_openeye_sdf(posed_mol, str(out_fn))

        # test reversed order
        dus.reverse()
        # Use posit to dock against each DU
        print("Running docking for all DUs")
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
        out_fn = out_dir / "docked_reversed.sdf"
        save_openeye_sdf(posed_mol, str(out_fn))

        print("Done")
