import asapdiscovery.ml
import mtenn
import numpy as np
import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.inference import GATInference, SchnetInference
from numpy.testing import assert_allclose


@pytest.fixture()
def docked_structure_file(scope="session"):
    return fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb")


@pytest.mark.parametrize(
    "target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "MERS-CoV-Mpro"]
)
def test_gatinference_construct_by_latest(target):
    inference_cls = GATInference.from_latest_by_target(target)
    assert inference_cls is not None
    assert inference_cls.model_type == "GAT"
    assert target in inference_cls.targets


def test_gatinference_construct_from_name(
    tmp_path,
):
    inference_cls = GATInference.from_model_name("gat_test_v0", local_dir=tmp_path)
    assert inference_cls is not None
    assert inference_cls.local_model_spec.local_dir == tmp_path


def test_gatinference_predict(test_data):
    inference_cls = GATInference.from_model_name("gat_test_v0")
    g1, _, _, _ = test_data
    assert inference_cls is not None
    output = inference_cls.predict(g1)
    assert output is not None


@pytest.mark.parametrize(
    "target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "MERS-CoV-Mpro"]
)
def test_gatinference_predict_smiles_equivariant(test_data, target):
    inference_cls = GATInference.from_latest_by_target(target)
    g1, g2, _, _ = test_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    assert_allclose(output1, output2, rtol=1e-5)


# test inference dataset cls against training dataset cls
@pytest.mark.parametrize(
    "target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "MERS-CoV-Mpro"]
)
def test_gatinference_predict_dataset(test_data, test_inference_data, target):
    inference_cls = GATInference.from_latest_by_target(target)
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


@pytest.mark.parametrize(
    "target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "MERS-CoV-Mpro"]
)
def test_gatinference_predict_from_smiles_dataset(
    test_data, test_inference_data, target
):
    inference_cls = GATInference.from_latest_by_target(target)
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
    inference_cls = GATInference.from_latest_by_target("SARS-CoV-2-Mpro")

    _, _, _, gids = test_inference_data
    gids_subset = gids[0:2:1]
    for g in gids_subset:
        res = inference_cls.predict(g)
        assert res


def test_schnet_inference_construct():
    inference_cls = SchnetInference.from_latest_by_target("SARS-CoV-2-Mpro")
    assert inference_cls is not None
    assert inference_cls.model_type == "schnet"
    assert type(inference_cls.model.readout) is mtenn.model.PIC50Readout


def test_schnet_inference_predict_from_structure_file(docked_structure_file):
    inference_cls = SchnetInference.from_latest_by_target("SARS-CoV-2-Mpro")
    assert inference_cls is not None
    output = inference_cls.predict_from_structure_file(docked_structure_file)
    assert output is not None


def test_schnet_inference_predict_from_pose(docked_structure_file):
    inference_cls = SchnetInference.from_latest_by_target("SARS-CoV-2-Mpro")

    dataset = asapdiscovery.ml.dataset.DockedDataset(
        [docked_structure_file], [("Mpro-P0008_0A", "ERI-UCB-ce40166b-17")]
    )
    assert inference_cls is not None
    c, pose = dataset[0]
    output = inference_cls.predict(pose)
    assert output is not None
