"""Subsample a dataset to a DockedDataset to the first N entries given size."""

import pickle
import numpy as np
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset
from asapdiscovery.ml.utils import split_dataset
from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate, ExperimentalCompoundData
from dgllife.utils import CanonicalAtomFeaturizer


def load_data(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

data = load_data("./achiral_enantiopure_best_02_13_23_dd.pkl")

print(data)

compound, pose = data[0]
print(pose)
print(pose.keys())

for k, v in pose.items():
    print(k, v, type(v))

print( compound[0], type(compound[0]))
print( compound[1], type(compound[1]))

data = {'compound_id':compound[1], 'smiles':pose['smiles'], 'experimental_data':{'pIC50':pose['pIC50'], 'pIC50_range':pose['pIC50_range'], 'pIC50_stderr':pose['pIC50_stderr']  }}

ecd = ExperimentalCompoundData(**data)

gds = GraphDataset([ecd], cache_file="./cache.bin", node_featurizer=CanonicalAtomFeaturizer())

print(ecd)
print(gds)
print(gds[0])
print(gds[0][1]["g"])
print(type(gds[0][1]["g"]))


# dump val to a pickle file 
with open("graph_ds.pkl", "wb") as f:
    pickle.dump(gds, f)
