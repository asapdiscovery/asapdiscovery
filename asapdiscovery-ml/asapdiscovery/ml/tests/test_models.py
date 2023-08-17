import os

import pytest
from asapdiscovery.ml.models import (
    ASAPMLModelRegistry,
    LocalMLModelSpec,
    MLModelRegistry,
    MLModelSpec,
)


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    return weights


def test_default_registry():
    assert ASAPMLModelRegistry.models != {}
    assert ASAPMLModelRegistry.models["gat_test_v0"].type == "GAT"


def test_pull_model():
    model = ASAPMLModelRegistry.models["gat_test_v0"]
    assert type(model) is MLModelSpec
    pulled_model = model.pull()
    assert type(pulled_model) is LocalMLModelSpec
    assert pulled_model.type == "GAT"


def test_pull_to_local_dir(tmp_path):
    model = ASAPMLModelRegistry.models["gat_test_v0"]
    assert type(model) is MLModelSpec
    local_model = model.pull(local_dir=tmp_path)
    assert os.path.exists(os.path.join(tmp_path, local_model.weights_file))
    assert os.path.exists(os.path.join(tmp_path, local_model.config_file))


@pytest.mark.parametrize(
    "target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "MERS-CoV-Mpro"]
)
def test_default_registry_targets(target):
    models_for_target = ASAPMLModelRegistry.get_models_for_target(target)
    assert len(models_for_target) > 0
    for model in models_for_target:
        assert target in model.targets


@pytest.mark.parametrize(
    "target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "MERS-CoV-Mpro"]
)
@pytest.mark.parametrize("type", ["GAT", "schnet"])
def test_default_registry_target_and_type(target, type):
    models_for_target_and_type = ASAPMLModelRegistry.get_models_for_target_and_type(
        target, type
    )
    for model in models_for_target_and_type:
        assert target in model.targets
        assert model.type == type


def test_get_model():
    model = ASAPMLModelRegistry.get_model("gat_test_v0")
    assert model.type == "GAT"


def test_get_latest_model_for_target_and_type():
    model = ASAPMLModelRegistry.get_latest_model_for_target_and_type(
        "SARS-CoV-2-Mpro", "GAT"
    )
    other_models = ASAPMLModelRegistry.get_models_for_target("SARS-CoV-2-Mpro")
    # sort by date updated
    other_models.sort(key=lambda x: x.last_updated, reverse=True)
    assert model == other_models[0]


def test_custom_registry(weights_yaml):
    registry = MLModelRegistry.from_yaml(weights_yaml)
    assert registry.models != {}
    assert registry.models["gatmodel_test"].type == "GAT"


def test_custom_registry_pull(weights_yaml):
    registry = MLModelRegistry.from_yaml(weights_yaml)
    model = registry.models["gatmodel_test"]
    assert type(model) is MLModelSpec
    pulled_model = model.pull()
    assert type(pulled_model) is LocalMLModelSpec
    assert pulled_model.type == "GAT"
