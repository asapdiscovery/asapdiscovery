import pytest
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import DockingInputPair
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.docking_v2 import POSITDocker


@pytest.fixture(scope="session")
def docking_data():
    sdf_file = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    pdb_file = fetch_test_file(
        "Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb"
    )
    ligand = Ligand.from_sdf(sdf_file, compound_name="test")
    cmplx = Complex.from_pdb(
        pdb_file,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    prepped_complex = PreppedComplex.from_complex(cmplx)
    return ligand, prepped_complex


@pytest.fixture(scope="session")
def docking_input_pair(docking_data):
    ligand, prepped_complex = docking_data
    return DockingInputPair(complex=prepped_complex, ligand=ligand)


def test_docking(docking_input_pair):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair])
    assert len(results) == 1

def test_docking_multiple(docking_input_pair):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair, docking_input_pair, docking_input_pair])
    assert len(results) == 3


def test_docking_with_file_write(docking_input_pair, tmp_path):
    docker = POSITDocker(write_files=True, output_dir=tmp_path)
    results = docker.dock([docking_input_pair])
