"""Convert a DockedDataset to a GraphDataset with just a few entries for testing purposes."""

import pickle

from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.ml.dataset import GraphDataset
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem

# lets make a synthetic dataset using a smiles from Fragalysis

# we will use a smiles from P0045_0A:TRY-UNI-2EDDB1FF-7 Fragalysis ID
input_smiles = "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(O[C@H]2CC(=O)N2)c1"

mol = Chem.MolFromSmiles(input_smiles)
# randomly shuffle the smiles string to test equivariance
reorder_smiles = Chem.MolToSmiles(mol, doRandom=True)


data = {
    "compound_id": "c1",
    "smiles": input_smiles,
    "experimental_data": {
        "pIC50": 1,
        "pIC50_range": 1,
        "pIC50_stderr": 1,
    },
}
data_reorder = {
    "compound_id": "c2",
    "smiles": reorder_smiles,
    "experimental_data": {
        "pIC50": 1,
        "pIC50_range": 1,
        "pIC50_stderr": 1,
    },
}


ecd = ExperimentalCompoundData(**data)
ecd_reorder = ExperimentalCompoundData(**data_reorder)

gds = GraphDataset(
    [ecd, ecd_reorder],
    cache_file="./cache.bin",
    node_featurizer=CanonicalAtomFeaturizer(),
)

print(ecd)
print(ecd_reorder)
print(gds)


# dump data to a pickle file
with open("fragalysis_GAT_test_ds.pkl", "wb") as f:
    pickle.dump(gds, f)
