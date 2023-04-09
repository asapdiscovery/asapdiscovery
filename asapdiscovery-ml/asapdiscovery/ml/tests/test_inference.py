import os
import pickle
import shutil

import asapdiscovery.ml
import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    # use to clean up in weights in
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    yield weights
    shutil.rmtree("./_weights", ignore_errors=True)


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
    g1, _, _, _ = test_data
    assert inference_cls is not None
    output = inference_cls.predict(g1)
    assert output is not None


def test_gatinference_predict_smiles_equivariant(weights_yaml, test_data):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    g1, g2, _, _ = test_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    assert_allclose(output1, output2)


# test inference dataset cls against training dataset cls
def test_gatinference_predict_dataset(weights_yaml, test_data, test_inference_data):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    g1, g2, g3, _ = test_data
    g1_infds, g2_infds, g3_infds, _ = test_inference_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    output3 = inference_cls.predict(g3)
    assert_allclose(output1, output2)

    # test inference dataset
    output_infds_1 = inference_cls.predict(g1_infds)
    output_infds_2 = inference_cls.predict(g2_infds)
    output_infds_3 = inference_cls.predict(g3_infds)
    assert_allclose(output_infds_1, output_infds_2)

    # test that the ones that should be the same are
    assert_allclose(output1, output_infds_1)
    assert_allclose(output2, output_infds_2)
    assert_allclose(output3, output_infds_3)

    # test that the ones that should be different are
    assert not np.allclose(output3, output1)
    assert not np.allclose(output3, output2)


def test_gatinference_predict_from_smiles_dataset(
    weights_yaml, test_data, test_inference_data
):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )
    g1, g2, _, _ = test_data
    g1_infds, g2_infds, _, gids = test_inference_data
    # same data different smiles order
    assert inference_cls is not None
    smiles = list(gids.smiles_dict.keys())
    s1 = smiles[0]
    s2 = smiles[1]
    # smiles one and two are the same
    output1 = inference_cls.predict(gids[s1])
    output2 = inference_cls.predict(gids[s2])
    assert_allclose(output1, output2)

    # smiles one and three are different
    s3 = smiles[2]
    output3 = inference_cls.predict(gids[s3])
    assert not np.allclose(output3, output1)

    # test predicting directly from smiles
    output_smiles_1 = inference_cls.predict_from_smiles(s1)
    output_smiles_2 = inference_cls.predict_from_smiles(s2)
    output_smiles_3 = inference_cls.predict_from_smiles(s3)

    assert_allclose(output_smiles_1, output_smiles_2)
    assert_allclose(output1, output_smiles_1)

    assert_allclose(output3, output_smiles_3)
    assert not np.allclose(output_smiles_3, output_smiles_1)
    assert not np.allclose(output3, output_smiles_1)

    # test predicting list of similes
    output_arr = inference_cls.predict_from_smiles([s1, s2, s3])
    assert_allclose(
        output_arr, np.asarray([output_smiles_1, output_smiles_2, output_smiles_3])
    )


def test_gatinference_predict_from_subset(weights_yaml, test_data, test_inference_data):
    inference_cls = asapdiscovery.ml.inference.GATInference(
        "gatmodel_test", weights_yaml
    )

    _, _, _, gids = test_inference_data
    gids_subset = gids[0:2:1]
    for g in gids_subset:
        res = inference_cls.predict(g)
        assert res
