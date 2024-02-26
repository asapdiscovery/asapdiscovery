from rdkit import Chem


def rdkit_smiles_roundtrip(smi: str) -> str:
    """
    Roundtrip a SMILES string through RDKit to canonicalize it

    Parameters
    ----------
    smi : str
        SMILES string to canonicalize

    Returns
    -------
    str
        Canonicalized SMILES string
    """
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol)
