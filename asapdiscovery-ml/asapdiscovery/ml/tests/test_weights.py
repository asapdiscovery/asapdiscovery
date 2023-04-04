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
    shutil.rmtree("./_hot_tub_time_machine", ignore_errors=True)


@pytest.mark.parametrize("force_fetch", [True, False])
@pytest.mark.parametrize(
    "path, should_raise",
    [
        (None, True),
        (False, True),
        ("./_weights/", False),
        ("_weights", False),
        ("_hot_tub_time_machine", False),
    ],
)
def test_fetch_weights(weights_yaml, force_fetch, path, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            _ = asapdiscovery.ml.weights.fetch_model_from_spec(
                weights_yaml, "model1", local_dir=path, force_fetch=force_fetch
            )
    else:
        specs = asapdiscovery.ml.weights.fetch_model_from_spec(
            weights_yaml,
            ["model1", "model2"],
            local_dir=path,
            force_fetch=force_fetch,
        )
        print(specs)
        # type
        assert specs["model1"].type == "GAT"
        assert specs["model2"].type == "blah"
        # config
        assert specs["model1"].config is None
        assert specs["model2"].config.exists()
        # weights
        assert specs["model1"].weights.exists()
        assert specs["model2"].weights.exists()

        # now fetch just one model that is already fetched, should not fetch again
        specs = asapdiscovery.ml.weights.fetch_model_from_spec(
            weights_yaml, "model1", local_dir=path, force_fetch=force_fetch
        )
        assert specs["model1"].weights.exists()
