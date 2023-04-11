from pathlib import Path

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    load_openeye_sdf,
    load_openeye_sdfs,
    oechem,
    oedocking,
    oespruce,
    save_openeye_pdb,
    save_openeye_sdf,
)
from asapdiscovery.docking import run_docking_oe


def test_loading_inputs():
    # Make output directory
    output_dir = Path("outputs/multireceptor_docking_test")
    if not output_dir.exists():
        output_dir.mkdir()

    # Load molecules
    ligand_sdf = "inputs/multireceptor_docking_test/docked.sdf"
    mol = load_openeye_sdf(ligand_sdf)

    # Load design units
    du_glob = "inputs/multireceptor_docking_test/*.oedu"
    du_files = list(Path().glob(du_glob))

    dus = []
    for du_fn in du_files:
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(du_fn), du)
        dus.append(du)

    print(
        f"Loaded {len(dus)} proteins from {du_glob} and {mol.GetTitle()} ligands from {ligand_sdf}"
    )
    return dus, mol, output_dir


def test_single_receptor_docking():
    dus, mol, out_dir = test_loading_inputs()
    # Use posit to dock against each DU
    print("Running docking for all DUs")
    for du in dus:
        success, posed_mol, docking_id = run_docking_oe(
            design_units=[du],
            orig_mol=mol,
            dock_sys="posit",
            relax="clash",
            hybrid=False,
            compound_name=mol.GetTitle(),
            use_omega=True,
            num_poses=1,
        )
        out_fn = out_dir / f"{du.GetTitle()}_docked.sdf"
        save_openeye_sdf(posed_mol, str(out_fn))

    print("Done")


def test_multireceptor_docking():
    dus, mol, out_dir = test_loading_inputs()
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
