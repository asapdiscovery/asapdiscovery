from pathlib import Path

import pytest
from asapdiscovery.data.schema_v2.molfile import MolFileFactory
from asapdiscovery.data.testing.test_resources import fetch_test_file
from pydantic import ValidationError


@pytest.fixture(scope="session")
def sdf_file():
    return fetch_test_file("Mpro_combined_labeled.sdf")


@pytest.fixture(scope="session")
def smi_file():
    return fetch_test_file("Mpro_combined_labeled.smi")


def test_molfile_factory_sdf(sdf_file):
    molfile = MolFileFactory.from_file(filename=sdf_file)
    assert len(molfile.ligands) == 576


def test_molfile_factory_smi(smi_file):
    molfile = MolFileFactory.from_file(filename=smi_file)
    assert len(molfile.ligands) == 556
