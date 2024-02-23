import asapdiscovery.ml
import mtenn
import numpy as np
import pytest
from asapdiscovery.data.backend.openeye import load_openeye_pdb
import torch
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
    inference_cls = GATInference.from_model_name(
        "asapdiscovery-SARS-CoV-2-Mpro-GAT-2024.02.06", local_dir=tmp_path
    )
    assert inference_cls is not None
    assert inference_cls.local_model_spec.local_dir == tmp_path


def test_gatinference_weights(tmp_path):
    inference_cls = GATInference.from_model_name(
        "asapdiscovery-SARS-CoV-2-Mpro-GAT-2024.02.06", local_dir=tmp_path
    )
    wts_file_params = torch.load(
        inference_cls.local_model_spec.weights_file,
        map_location=inference_cls.device,
    )

    param_mismatches = []
    for k, model_param in inference_cls.model.named_parameters():
        if not torch.allclose(model_param, wts_file_params[k]):
            param_mismatches.append(k)

    assert len(param_mismatches) == 0, param_mismatches


def test_gatinference_predict(test_data):
    inference_cls = GATInference.from_model_name(
        "asapdiscovery-SARS-CoV-2-Mpro-GAT-2024.02.06"
    )
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
def test_gatinference_predict_dataset(test_data, target):
    inference_cls = GATInference.from_latest_by_target(target)
    g1, g2, g3, _ = test_data
    # same data different smiles order
    assert inference_cls is not None
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    output3 = inference_cls.predict(g3)
    assert_allclose(output1, output2, rtol=1e-5)

    # test that the ones that should be different are
    assert not np.allclose(output3, output1, rtol=1e-5)
    assert not np.allclose(output3, output2, rtol=1e-5)


@pytest.mark.parametrize(
    "target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "MERS-CoV-Mpro"]
)
def test_gatinference_predict_from_smiles_dataset(test_data, target):
    inference_cls = GATInference.from_latest_by_target(target)
    g1, g2, g3, gds = test_data
    # same data different smiles order
    assert inference_cls is not None
    smiles = [pose["smiles"] for _, pose in gds]
    s1 = smiles[0]
    s2 = smiles[1]
    # smiles one and two are the same
    output1 = inference_cls.predict(g1)
    output2 = inference_cls.predict(g2)
    assert_allclose(output1, output2, rtol=1e-5)

    # smiles one and three are different
    s3 = smiles[2]
    output3 = inference_cls.predict(g3)
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

    # test predicting list of smiles
    output_arr = inference_cls.predict_from_smiles([s1, s2, s3])
    assert_allclose(
        output_arr,
        np.asarray([output_smiles_1, output_smiles_2, output_smiles_3]),
    )


def test_gatinference_predict_from_subset(test_data):
    inference_cls = GATInference.from_latest_by_target("SARS-CoV-2-Mpro")

    _, _, _, gids = test_data
    gids_subset = gids[0:2:1]
    for _, g in gids_subset:
        res = inference_cls.predict(g["g"])
        assert res


def test_schnet_inference_construct():
    inference_cls = SchnetInference.from_latest_by_target("SARS-CoV-2-Mpro")
    assert inference_cls is not None
    assert inference_cls.model_type == "schnet"
    assert type(inference_cls.model.readout) is mtenn.readout.PIC50Readout


def test_schnet_inference_weights(tmp_path):
    inference_cls = SchnetInference.from_model_name(
        "asapdiscovery-SARS-CoV-2-Mpro-schnet-2024.02.05", local_dir=tmp_path
    )
    wts_file_params = torch.load(
        inference_cls.local_model_spec.weights_file,
        map_location=inference_cls.device,
    )

    param_mismatches = []
    for k, model_param in inference_cls.model.named_parameters():
        if not torch.allclose(model_param, wts_file_params[k]):
            param_mismatches.append(k)

    assert len(param_mismatches) == 0, param_mismatches


def test_schnet_inference_predict_from_structure_file(docked_structure_file):
    inference_cls = SchnetInference.from_latest_by_target("SARS-CoV-2-Mpro")
    assert inference_cls is not None
    output = inference_cls.predict_from_structure_file(docked_structure_file)
    assert output is not None


def test_schnet_inference_predict_from_pose(docked_structure_file):
    inference_cls = SchnetInference.from_latest_by_target("SARS-CoV-2-Mpro")

    dataset = asapdiscovery.ml.dataset.DockedDataset.from_files(
        str_fns=[docked_structure_file],
        compounds=[("Mpro-P0008_0A", "ERI-UCB-ce40166b-17")],
    )
    assert inference_cls is not None
    c, pose = dataset[0]
    output = inference_cls.predict(pose)
    assert output is not None


def test_schnet_inference_predict_from_oemol(docked_structure_file):
    inference_cls = SchnetInference.from_latest_by_target("SARS-CoV-2-Mpro")

    pose_oemol = load_openeye_pdb(docked_structure_file)
    assert inference_cls is not None
    output = inference_cls.predict_from_oemol(pose_oemol)
    assert output is not None
