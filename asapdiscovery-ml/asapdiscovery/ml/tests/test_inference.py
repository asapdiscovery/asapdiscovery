import os
import shutil

import asapdiscovery.ml
import pytest


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    yield weights
    shutil.rmtree("./_weights", ignore_errors=True)


@pytest.fixture()
def test_data():
    return None
    # test data


def test_gatinference_construct(weights_yaml):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    assert inference_cls is not None

def test_inference_construct_no_spec(weights_yaml):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "model1"
    )
    assert inference_cls is not None

def test_gatinference_predict(weights_yaml, test_data):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    assert inference_cls is not None
    # output = inference_cls.predict(test_data)
    # assert output is not None
