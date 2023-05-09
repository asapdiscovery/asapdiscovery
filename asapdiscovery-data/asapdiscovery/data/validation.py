
from asapdiscovery.data.openeye import oechem
import warnings

def read_file_as_str(filename):
    with open(filename, "r") as f:
        return f.read()

def write_file_from_string(filename):
    with open(filename, "w") as f:
        f.write(string)

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

def sdf_length(sdf_fn):
    if not Path(sdf_fn).exists():
        raise FileNotFoundError(f"{sdf_fn} does not exist!")
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    if ifs.open(sdf_fn):
        nmols = len(ifs.GetOEGraphMols())
        return nmols
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")
    
def is_single_molecule_sdf(sdf_fn):
    sdf_len = sdf_length(sdf_fn)
    if sdf_len > 1:
        return False
    elif sdf_len == 0:
        raise ValueError("SDF file must contain at least one molecule")
    else:
        return True

def is_multiligand_sdf(sdf_fn):
    return not is_single_molecule_sdf(sdf_fn)

