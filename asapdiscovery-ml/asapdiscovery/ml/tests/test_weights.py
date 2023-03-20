import shutil
import os
import asapdiscovery.ml
import pytest


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    yield weights
    shutil.rmtree("./_weights", ignore_errors=True)


@pytest.mark.parametrize("force_fetch", [True, False])
def test_fetch_weights(weights_yaml, force_fetch):
    _ = asapdiscovery.ml.weights.fetch_weights_from_spec(
        weights_yaml,
        ["schnet", "schnet2"],
        local_path="./_weights/",
        force_fetch=force_fetch,
    )
    # now fetch just one model that is already fetched, should not fetch again
    _ = asapdiscovery.ml.weights.fetch_weights_from_spec(
        weights_yaml, "schnet", local_path="./_weights/", force_fetch=force_fetch
    )
