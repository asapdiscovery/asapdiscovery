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


def test_load_from_sdf(sdf_file):
    mols = load_exp_from_sdf(str(sdf_file))
    assert len(mols) == 576
