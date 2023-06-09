import os
import shutil

import asapdiscovery.ml
import pytest


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    return weights

@pytest.fixture()
def outputs(tmp_path):
    """Creates outputs directory in temp location and returns path"""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    yield outputs
    shutil.rmtree(outputs)


@pytest.mark.parametrize("force_fetch", [True, False])
def test_fetch_weights(weights_yaml, force_fetch, outputs):
    print(outputs)
    specs = asapdiscovery.ml.weights.fetch_model_from_spec(
        weights_yaml,
        ["model1", "model2"],
        local_dir=outputs,
        force_fetch=force_fetch,
    )
    # type
    assert specs["model1"].type == "GAT"
    assert specs["model2"].type == "blah"
    # config
    assert specs["model1"].config is None
    assert specs["model2"].config.exists()
    # weights
    assert specs["model1"].weights.exists()
    assert specs["model2"].weights.exists()


@pytest.mark.parametrize("force_fetch", [True, False])
@pytest.mark.parametrize("path", [None, False])
def test_fetch_weights_invalid_path(weights_yaml, force_fetch, outputs):
    with pytest.raises(ValueError):
        _ = asapdiscovery.ml.weights.fetch_model_from_spec(
            weights_yaml, "model1", local_dir=path, force_fetch=force_fetch
        )