import os

from asapdiscovery.data.metadata.resources import (
    MERS_CoV_Mpro_SEQRES,
    SARS_CoV_2_Mac1_SEQRES,
    SARS_CoV_2_Mpro_SEQRES,
)


def test_seqres():
    assert os.path.exists(MERS_CoV_Mpro_SEQRES)
    assert os.path.exists(SARS_CoV_2_Mpro_SEQRES)
    assert os.path.exists(SARS_CoV_2_Mac1_SEQRES)
