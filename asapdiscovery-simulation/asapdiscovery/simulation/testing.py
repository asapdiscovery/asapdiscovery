# TODO: Add tests that can use a ligand as well
def test_forcefield_generation(input_pdb_path: str):
    from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile
    from openmm.unit import amu, nanometers

    # Input Files
    pdb = PDBFile(input_pdb_path)
    forcefield = ForceField(
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    )

    # System Configuration

    nonbondedMethod = PME
    nonbondedCutoff = 1.0 * nanometers
    ewaldErrorTolerance = 0.000001
    constraints = HBonds
    rigidWater = True
    hydrogenMass = 4.0 * amu

    # Prepare the Simulation
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, padding=0.9 * nanometers, model="tip3p")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=nonbondedMethod,
        nonbondedCutoff=nonbondedCutoff,
        constraints=constraints,
        rigidWater=rigidWater,
        ewaldErrorTolerance=ewaldErrorTolerance,
        hydrogenMass=hydrogenMass,
    )
    if system:
        return True
    else:
        return False
