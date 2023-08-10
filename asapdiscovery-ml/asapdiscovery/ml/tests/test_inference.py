import os
import shutil

import asapdiscovery.ml
import numpy as np
import pytest

from asapdiscovery.ml.inference import GATInference, SchnetInference
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

from numpy.testing import assert_allclose


@pytest.fixture()
def weights_yaml():
    # ugly hack to make the directory relative
    # use to clean up in weights in
    weights = os.path.join(os.path.dirname(__file__), "test_weights.yaml")
    return weights


@pytest.fixture()
def outputs(tmp_path):
    """Creates outputs directory in temp location and returns path"""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    yield outputs
    shutil.rmtree(outputs)


@pytest.fixture()
def docked_structure_file(scope="session"):
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb")


def test_gatinference_construct(outputs):
    inference_cls = GATInference(
        "SARS-CoV-2-Mpro", model_name="gat_test_v0", weights_local_dir=outputs
    )
    assert inference_cls is not None
    assert inference_cls.model_type == "GAT"


def test_inference_construct_no_spec(outputs):
    inference_cls = GATInference(weights_local_dir=outputs)
    assert inference_cls is not None


def test_gatinference_predict(test_data, outputs):
    inference_cls = GATInference(
        "SARS-CoV-2-Mpro", model_name="gat_test_v0", weights_local_dir=outputs
    )
    g1, _, _, _ = test_data
    assert inference_cls is not None
    output = inference_cls.predict(g1)
    assert output is not None


def test_gatinference_predict_smiles_equivariant(test_data, outputs):
    inference_cls = GATInference(
        "SARS-CoV-2-Mpro", "gat_test_v0", weights_local_dir=outputs
    )
    g1, g2, _, _ = test_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    assert_allclose(output1, output2, rtol=1e-5)


# test inference dataset cls against training dataset cls
def test_gatinference_predict_dataset(test_data, test_inference_data):
    inference_cls = GATInference("SARS-CoV-2-Mpro", model_name="gat_test_v0")
    g1, g2, g3, _ = test_data
    g1_infds, g2_infds, g3_infds, _ = test_inference_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    output3 = inference_cls.predict(g3)
    assert_allclose(output1, output2, rtol=1e-5)

    # test inference dataset
    output_infds_1 = inference_cls.predict(g1_infds)
    output_infds_2 = inference_cls.predict(g2_infds)
    output_infds_3 = inference_cls.predict(g3_infds)
    assert_allclose(output_infds_1, output_infds_2, rtol=1e-5)

    # test that the ones that should be the same are
    assert_allclose(output1, output_infds_1, rtol=1e-5)
    assert_allclose(output2, output_infds_2, rtol=1e-5)
    assert_allclose(output3, output_infds_3, rtol=1e-5)

    # test that the ones that should be different are
    assert not np.allclose(output3, output1, rtol=1e-5)
    assert not np.allclose(output3, output2, rtol=1e-5)


def test_gatinference_predict_from_smiles_dataset(test_data, test_inference_data):
    inference_cls = GATInference("SARS-CoV-2-Mpro", model_name="gatmodel_test")
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
    assert_allclose(output1, output2, rtol=1e-5)

    # smiles one and three are different
    s3 = smiles[2]
    output3 = inference_cls.predict(gids[s3])
    assert not np.allclose(output3, output1, rtol=1e-5)

    # test predicting directly from smiles
    output_smiles_1 = inference_cls.predict_from_smiles(s1)
    output_smiles_2 = inference_cls.predict_from_smiles(s2)
    output_smiles_3 = inference_cls.predict_from_smiles(s3)

    assert_allclose(output_smiles_1, output_smiles_2, rtol=1e-5)
    assert_allclose(output1, output_smiles_1, rtol=1e-5)

    assert_allclose(output3, output_smiles_3, rtol=1e-5)
    assert not np.allclose(output_smiles_3, output_smiles_1, rtol=1e-5)
    assert not np.allclose(output3, output_smiles_1, rtol=1e-5)

    # test predicting list of similes
    output_arr = inference_cls.predict_from_smiles([s1, s2, s3])
    assert_allclose(
        output_arr,
        np.asarray([output_smiles_1, output_smiles_2, output_smiles_3]),
    )


def test_gatinference_predict_from_subset(test_inference_data):
    inference_cls = GATInference("SARS-CoV-2-Mpro", model_name="gat_test_v0")

    _, _, _, gids = test_inference_data
    gids_subset = gids[0:2:1]
    for g in gids_subset:
        res = inference_cls.predict(g)
        assert res


def test_schnet_inference_construct():
    inference_cls = SchnetInference(
        "SARS-CoV-2-Mpro", model_name="asapdiscovery-schnet-2023.04.29"
    )
    assert inference_cls is not None
    assert inference_cls.model_type == "schnet"


def test_schnet_inference_predict_from_structure_file(docked_structure_file):
    inference_cls = SchnetInference(
        "SARS-CoV-2-Mpro", model_name="asapdiscovery-schnet-2023.04.29"
    )
    assert inference_cls is not None
    output = inference_cls.predict_from_structure_file(docked_structure_file)
    assert output is not None


def test_schnet_inference_predict_from_pose(docked_structure_file):
    inference_cls = SchnetInference("SARS-CoV-2-Mpro")

    dataset = asapdiscovery.ml.dataset.DockedDataset(
        [docked_structure_file], [("Mpro-P0008_0A", "ERI-UCB-ce40166b-17")]
    )
    assert inference_cls is not None
    c, pose = dataset[0]
    output = inference_cls.predict(pose)
    assert output is not None
