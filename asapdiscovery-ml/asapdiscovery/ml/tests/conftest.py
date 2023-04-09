import os
import shutil
import pytest

from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.ml.dataset import GraphDataset, GraphInferenceDataset
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem


# lets make a synthetic dataset using a smiles from Fragalysis
@pytest.fixture()
def fragalysis_smiles_1():
    # we will use a smiles from P0045_0A:TRY-UNI-2EDDB1FF-7 Fragalysis ID for compound 1 and reordered version
    return "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(O[C@H]2CC(=O)N2)c1"


@pytest.fixture()
def fragalysis_smiles_2():
    # for compound 2 we will use a different smiles P1073_0A:MAT-POS-E6DD326D-8 Fragalysis ID
    return "C=CC(=O)NC[C@@]1(C(=O)Nc2cncc3ccccc23)CCOc2ccc(Cl)cc21"


@pytest.fixture()
def fragalysis_smiles_1_reordered(fragalysis_smiles_1):
    mol = Chem.MolFromSmiles(fragalysis_smiles_1)
    # randomly shuffle the smiles string to test equivariance
    reorder_smiles = Chem.MolToSmiles(mol, doRandom=True)
    return reorder_smiles


@pytest.fixture()
def experimental_compound_data_1(fragalysis_smiles_1):
    data = {
        "compound_id": "c1",
        "smiles": fragalysis_smiles_1,
        "experimental_data": {
            "pIC50": 1,
            "pIC50_range": 1,
            "pIC50_stderr": 1,
        },
    }
    return ExperimentalCompoundData(**data)


@pytest.fixture()
def experimental_compound_data_2(fragalysis_smiles_2):
    data = {
        "compound_id": "c2",
        "smiles": fragalysis_smiles_2,
        "experimental_data": {
            "pIC50": 1,
            "pIC50_range": 1,
            "pIC50_stderr": 1,
        },
    }
    return ExperimentalCompoundData(**data)


@pytest.fixture()
def experimental_compound_data_1_reordered(fragalysis_smiles_1_reordered):
    data = {
        "compound_id": "c1",
        "smiles": fragalysis_smiles_1_reordered,
        "experimental_data": {
            "pIC50": 1,
            "pIC50_range": 1,
            "pIC50_stderr": 1,
        },
    }
    return ExperimentalCompoundData(**data)


@pytest.fixture()
def graph_dataset(
    experimental_compound_data_1,
    experimental_compound_data_1_reordered,
    experimental_compound_data_2,
):
    gds = GraphDataset(
        [
            experimental_compound_data_1,
            experimental_compound_data_1_reordered,
            experimental_compound_data_2,
        ],
        cache_file="./cache.bin",
        node_featurizer=CanonicalAtomFeaturizer(),
    )
    yield gds
    # clean up the cache file
    if os.path.exists("./cache.bin"):
        os.remove("./cache.bin")


@pytest.fixture()
def graph_inference_dataset(
    experimental_compound_data_1,
    experimental_compound_data_1_reordered,
    experimental_compound_data_2,
):
    gds = GraphInferenceDataset(
        [
            experimental_compound_data_1,
            experimental_compound_data_1_reordered,
            experimental_compound_data_2,
        ],
        cache_file="./cache.bin",
        node_featurizer=CanonicalAtomFeaturizer(),
    )
    yield gds
    # clean up the cache file
    if os.path.exists("./cache.bin"):
        os.remove("./cache.bin")


@pytest.fixture()
def test_data(graph_dataset):
    # contains three data points in a GraphDataset, first two same smiles reordered in the second one
    # and the third one is a different smiles

    # has structure: ((design_unit, compound),  {smiles: smiles, g: graph, **kwargs})
    # we want the graph
    g1 = graph_dataset[0][1]["g"]
    g2 = graph_dataset[1][1]["g"]
    g3 = graph_dataset[2][1]["g"]
    return g1, g2, g3, graph_dataset


@pytest.fixture()
def test_inference_data(graph_inference_dataset):
    # has structure: graph
    g1 = graph_inference_dataset[0]
    g2 = graph_inference_dataset[1]
    g3 = graph_inference_dataset[2]
    return g1, g2, g3, graph_inference_dataset
