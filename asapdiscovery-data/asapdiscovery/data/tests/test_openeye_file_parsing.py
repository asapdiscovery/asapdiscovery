from asapdiscovery.data.utils import (
    load_exp_from_sdf,
    oe_load_exp_from_file,
    exp_data_to_oe_mols,
)
from asapdiscovery.data.testing.test_resources import fetch_test_file
import pytest


@pytest.fixture
def sdf_file():
    return fetch_test_file("Mpro_combined_labeled.sdf")


@pytest.fixture
def smi_file():
    return fetch_test_file("Mpro_combined_labeled.smi")


@pytest.fixture
def file_dict():
    return {
        "sdf": fetch_test_file("Mpro_combined_labeled.sdf"),
        "smi": fetch_test_file("Mpro_combined_labeled.smi"),
    }


def test_load_from_sdf(sdf_file):
    mols = load_exp_from_sdf(str(sdf_file))
    assert len(mols) == 576
    # This is the current behaviour, which I dislike. Should be possible to have it pull an id from the sdf file
    assert mols[0].compound_id == "0"


@pytest.mark.parametrize("ftype", ["sdf", "smi"])
@pytest.mark.parametrize("return_mols_from_oe_load_exp_from_file", [True, False])
def test_load_exp_from_file(file_dict, ftype, return_mols_from_oe_load_exp_from_file):
    input_file = file_dict[ftype]
    if return_mols_from_oe_load_exp_from_file:
        cmpds, mols = oe_load_exp_from_file(
            str(input_file), ftype, return_mols=return_mols_from_oe_load_exp_from_file
        )
    else:
        cmpds = oe_load_exp_from_file(
            str(input_file), ftype, return_mols=return_mols_from_oe_load_exp_from_file
        )
        mols = exp_data_to_oe_mols(cmpds)
    assert len(cmpds) == len(mols)

    # These are different because converting to smiles removes duplicates
    if ftype == "sdf":
        assert len(cmpds) == 576
        assert len(mols) == 576
        assert mols[0].GetTitle() == "ALP-POS-477dc5b7-2"
        assert cmpds[0].compound_id == "ALP-POS-477dc5b7-2"
    elif ftype == "smi":
        assert len(cmpds) == 556
        assert len(mols) == 556
        assert mols[0].GetTitle() == "AAR-POS-d2a4d1df-26"
        assert cmpds[0].compound_id == "AAR-POS-d2a4d1df-26"


# TODO: Add tests for smiles files without compound_ids
