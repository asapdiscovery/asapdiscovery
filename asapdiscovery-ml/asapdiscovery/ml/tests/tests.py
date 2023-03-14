import asapdiscovery.ml
import pytest
import shutil

@pytest.fixture()
def weights_yaml():
    weights = "test_weights.yaml"
    yield weights
    shutil.rmtree('./_weights', ignore_errors=True)

    

@pytest.mark.parametrize("force_fetch", [True, False])
def test_fetch_weights(weights_yaml, force_fetch):
    w = asapdiscovery.ml.weights.fetch_weights_from_spec(weights_yaml, ["schnet", "schnet2"], local_path="./_weights/", force_fetch=force_fetch)
    # now fetch just one model that is already fetched, should not fetch again
    w = asapdiscovery.ml.weights.fetch_weights_from_spec(weights_yaml, "schnet", local_path="./_weights/", force_fetch=force_fetch)


