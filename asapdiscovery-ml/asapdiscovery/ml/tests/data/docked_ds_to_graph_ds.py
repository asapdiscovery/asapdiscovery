"""Convert a DockedDataset to a GraphDataset with just a few entries for testing purposes."""

import pickle

import numpy as np
from asapdiscovery.data.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
)
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset
from asapdiscovery.ml.utils import split_dataset
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem


def load_data(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


data = load_data("./achiral_enantiopure_best_02_13_23_dd.pkl")

print(data)

# we will only take a single value
compound, pose = data[0]

# randomly shuffle the smiles string
mol = Chem.MolFromSmiles(pose["smiles"])
newsmiles = Chem.MolToSmiles(mol, doRandom=True)


data = {
    "compound_id": compound[1],
    "smiles": pose["smiles"],
    "experimental_data": {
        "pIC50": pose["pIC50"],
        "pIC50_range": pose["pIC50_range"],
        "pIC50_stderr": pose["pIC50_stderr"],
    },
}
data_reorder = {
    "compound_id": compound[1],
    "smiles": newsmiles,
    "experimental_data": {
        "pIC50": pose["pIC50"],
        "pIC50_range": pose["pIC50_range"],
        "pIC50_stderr": pose["pIC50_stderr"],
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


# dump val to a pickle file
with open("graph_ds.pkl", "wb") as f:
    pickle.dump(gds, f)
