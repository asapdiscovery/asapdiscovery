"""
Created on Sat Jan 28 18:10:15 2023
@author: kendalllemons
Edited by: Alex Payne
"""
from openmm.app import *
from openmm.unit import *
from openmm.app import PDBFile


def main(input_pdb_path):
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

    print("Building system...")
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
    print("Success!")


if __name__ == "__main__":

    ## Original Prepped P2660 structure
    try:
        main(
            "inputs/01_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_prepped_receptor_0.pdb"
        )
    except ValueError as error:
        print(f"Error was: {error}")
        if (
            "No template found for residue 307 (LIG)." in error.__str__()
        ):
            print("Expected Error")
        else:
            print("Unexpected Error")

    ## Fauxalysis output
    try:
        main("inputs/02_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_4RSP_fauxalysis.pdb")
    except ValueError as error:
        print(f"Error was: {error}")
        if ( "No template found for residue 145 (CSO).  The set of atoms matches CCYS, but the bonds are different." in error.__str__()
        ):
            print("Expected Error")
        else:
            print("Unexpected Error")

    ## After running perses prep script
    try:
        main(
            "inputs/03_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_4RSP_fauxalysis_protonated.pdb"
        )
    except ValueError as error:
        print(f"Error was: {error}")
        if ("No template found for residue 8 (HIS).  The set of atoms matches HIP, but the bonds are different." in error.__str__()
        ):
            print("Expected Error")
        else:
            print("Unexpected Error")

    ## ASAP Prepped
    try:
        main("inputs/prepped_receptor_0.pdb")
    except ValueError as error:
        print(f"Error was: {error}")
        if (
            "No template found for residue 303 (LIG)." in error.__str__()
        ):
            print("Expected Error")
        else:
            print("Unexpected Error")


    ## Openmm-Setup prepped
    main("inputs/prepped_receptor_0-processed.pdb")
