
from asapdiscovery.data.openeye import oechem

def is_valid_smiles(smiles):
    # Create an OEMol object
    mol = oechem.OEMol()

    # Attempt to parse the SMILES string
    if not oechem.OEParseSmiles(mol, smiles):
        return False

    # Check if the parsed molecule is valid
    if not mol.IsValid():
        return False

    return True