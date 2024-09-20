import os

import pytest
from asapdiscovery.ml.models import (
    ASAPMLModelRegistry,
    LocalMLModelSpec,
    MLModelBase,
    MLModelRegistry,
    MLModelSpec,
    RemoteEnsembleHelper,
)


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    return weights


def test_default_registry():
    assert ASAPMLModelRegistry.models != {}
    assert ASAPMLModelRegistry.models["gat_test"].type == "GAT"


def test_pull_model():
    model = ASAPMLModelRegistry.models["gat_test"]
    assert type(model) is MLModelSpec
    pulled_model = model.pull()
    assert type(pulled_model) is LocalMLModelSpec
    assert pulled_model.type == "GAT"


def test_pull_to_local_dir(tmp_path):
    model = ASAPMLModelRegistry.models["gat_test"]
    assert type(model) is MLModelSpec
    local_model = model.pull(local_dir=tmp_path)
    assert os.path.exists(os.path.join(tmp_path, local_model.weights_file))
    assert os.path.exists(os.path.join(tmp_path, local_model.config_file))


def test_default_registry_ensemble():
    # name comes from remote ensemble manifest
    assert ASAPMLModelRegistry.models["asapdiscovery-GAT-ensemble-test"].type == "GAT"


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
    model = ASAPMLModelRegistry.get_model("gat_test")
    assert model.type == "GAT"


@pytest.mark.parametrize("type", ["GAT", "schnet"])
def test_get_latest_model_for_target_and_type(type):
    model = ASAPMLModelRegistry.get_latest_model_for_target_and_type(
        "SARS-CoV-2-Mpro", type
    )
    other_models = ASAPMLModelRegistry.get_models_for_target("SARS-CoV-2-Mpro")
    # filter by type
    other_models = [m for m in other_models if m.type == type]
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


def test_remote_ensemble_pull(remote_ensemble_manifest_url):
    reh = RemoteEnsembleHelper(manifest_url=remote_ensemble_manifest_url)
    ens_mods = reh.to_ensemble_spec()
    emodspec = ens_mods["asapdiscovery-GAT-ensemble-test"]
    lemodspec = emodspec.pull()
    assert len(lemodspec.models) == 5


def test_registry_get_models_for_target_and_type():
    models = ASAPMLModelRegistry.get_models_for_target_and_type(
        "SARS-CoV-2-Mpro", "GAT"
    )
    assert isinstance(models, list)
    assert len(models) > 0
    for model in models:
        assert model.type == "GAT"
        assert "SARS-CoV-2-Mpro" in model.targets


def test_registry_get_models_for_target():
    models = ASAPMLModelRegistry.get_models_for_target("SARS-CoV-2-Mpro")
    assert len(models) > 0


def test_registry_get_targets_with_models():
    targets = ASAPMLModelRegistry.get_targets_with_models()
    assert len(targets) > 0
    assert "SARS-CoV-2-Mpro" in targets
    assert "SARS-CoV-2-Mac1" in targets
    assert "MERS-CoV-Mpro" in targets


def test_registry_get_latest_model_for_target_and_type():
    model = ASAPMLModelRegistry.get_latest_model_for_target_and_type(
        "SARS-CoV-2-Mpro", "GAT"
    )
    assert isinstance(model, MLModelBase)
    assert model.type == "GAT"


def test_registry_get_latest_model_for_target_and_endpoint():
    model = ASAPMLModelRegistry.get_latest_model_for_target_and_endpoint(
        "SARS-CoV-2-Mpro", "pIC50"
    )
    assert isinstance(model, MLModelBase)
    assert model.type == "GAT"


def test_registry_get_models_for_endpoint():
    models = ASAPMLModelRegistry.get_models_for_endpoint("pIC50")
    assert isinstance(models, list)
    assert len(models) > 0
    for model in models:
        assert model.endpoint == "pIC50"


def test_registry_get_latest_model_for_endpoint():
    model = ASAPMLModelRegistry.get_latest_model_for_endpoint("pIC50")
    assert isinstance(model, MLModelBase)
    assert model.endpoint == "pIC50"


def test_registry_get_models_without_target():
    models = ASAPMLModelRegistry.get_models_without_target()
    assert isinstance(models, list)
    assert len(models) > 0
    for model in models:
        assert model.targets == {None}


def test_registry_get_endpoints():
    endpoints = ASAPMLModelRegistry.get_endpoints()
    assert isinstance(endpoints, list)
    assert len(endpoints) > 0


def test_registry_get_endpoints_for_target():
    endpoints = ASAPMLModelRegistry.get_endpoints_for_target("SARS-CoV-2-Mpro")
    assert isinstance(endpoints, list)
    assert len(endpoints) > 0
    assert "pIC50" in endpoints


def test_registry_endpoint_has_target():
    assert ASAPMLModelRegistry.endpoint_has_target("pIC50")
    assert not ASAPMLModelRegistry.endpoint_has_target("LogD")


def test_registry_get_latest_model_for_target_type_and_endpoint():
    model = ASAPMLModelRegistry.get_latest_model_for_target_type_and_endpoint(
        "SARS-CoV-2-Mpro", "GAT", "pIC50"
    )
    assert isinstance(model, MLModelBase)
    assert model.type == "GAT"
    assert model.endpoint == "pIC50"


def test_registry_get_model_types_for_endpoint():
    types = ASAPMLModelRegistry.get_model_types_for_endpoint("pIC50")
    assert isinstance(types, list)
    assert len(types) > 0
    assert "GAT" in types


def test_registry_reccomend_models_for_target():
    models = ASAPMLModelRegistry.reccomend_models_for_target("SARS-CoV-2-Mpro")
    assert isinstance(models, list)
    assert len(models) > 0
    for model in models:
        if model.targets == {None}:
            continue
        assert "SARS-CoV-2-Mpro" in model.targets


def test_refresh_registry():
    # just check its functional
    prev_time = ASAPMLModelRegistry.time_updated
    ASAPMLModelRegistry.update_registry()
    assert ASAPMLModelRegistry.time_updated != prev_time
