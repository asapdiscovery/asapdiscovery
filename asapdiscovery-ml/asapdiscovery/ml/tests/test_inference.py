import os
import pickle
import shutil

import asapdiscovery.ml
import pytest
import torch
from numpy.testing import assert_allclose


def load_data(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    # use to clean up in weights in
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    yield weights
    shutil.rmtree("./_weights", ignore_errors=True)


@pytest.fixture()
def test_data():
    # ugly hack to make the directory relative
    # contains two data points in a GraphDataset, both the same with the smiles order changed in the second one
    data = load_data(
        os.path.join(os.path.dirname(__file__), "data/fragalysis_GAT_test_ds.pkl")
    )
    # has structure ((design_unit, compound),  {smiles: smiles, g: graph, **kwargs})
    # we want the graph
    g1 = data[0][1]["g"]
    g2 = data[1][1]["g"]
    return g1, g2


def test_gatinference_construct(weights_yaml):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    assert inference_cls is not None


def test_inference_construct_no_spec(weights_yaml):
    inference_cls = asapdiscovery.ml.inference.GATInference("model1")
    assert inference_cls is not None


def test_gatinference_predict(weights_yaml, test_data):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    assert inference_cls is not None
    output = inference_cls.predict(test_data)
    assert output is not None


def test_gatinference_predict(weights_yaml, test_data):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    g1, _ = test_data
    assert inference_cls is not None
    output = inference_cls.predict(g1)
    assert output is not None


def test_gatinference_predict_smiles_equivariant(weights_yaml, test_data):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    g1, g2 = test_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    assert_allclose(output1, output2)
