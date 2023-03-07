"""
Created on Sat Jan 28 18:10:15 2023
@author: kendalllemons
Edited by: Alex Payne
"""
from asapdiscovery.simulation.utils import test_forcefield_generation

if __name__ == "__main__":

    ## Original Prepped P2660 structure
    try:
        test_forcefield_generation(
            "inputs/01_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_prepped_receptor_0.pdb"
        )
    except ValueError as error:
        print(f"Error was: {error}")
        if (
            error.__str__()
            == "No template found for residue 307 (LIG).  This might mean your input topology is missing some atoms or bonds, or possibly that you are using the wrong force field."
        ):
            print("Expected Error")
        else:
            print("Unexpected Error")

    ## Fauxalysis output
    try:
        test_forcefield_generation(
            "inputs/02_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_4RSP_fauxalysis.pdb"
        )
    except ValueError as error:
        print(f"Error was: {error}")
        if (
            error.__str__()
            == "No template found for residue 145 (CSO).  The set of atoms matches CCYS, but the bonds are different."
        ):
            print("Expected Error")
        else:
            print("Unexpected Error")

    ## After running perses prep script
    try:
        test_forcefield_generation(
            "inputs/03_Mpro-P2660_0A_EDG-MED-b1ef7fe3-1_4RSP_fauxalysis_protonated.pdb"
        )
    except ValueError as error:
        print(f"Error was: {error}")
        if (
            error.__str__()
            == "No template found for residue 8 (HIS).  The set of atoms matches HIP, but the bonds are different."
        ):
            print("Expected Error")
        else:
            print("Unexpected Error")
