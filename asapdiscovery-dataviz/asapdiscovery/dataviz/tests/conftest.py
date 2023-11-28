import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def pose():
    pose = fetch_test_file("Mpro-P2660_0A_bound-prepped_ligand.sdf")
    return pose


@pytest.fixture(scope="session")
def protein():
    protein = fetch_test_file("Mpro-P2660_0A_bound-prepped_complex.pdb")
    return protein


@pytest.fixture(scope="session")
def top():
    top = fetch_test_file("example_traj_top.pdb")
    return top


@pytest.fixture(scope="session")
def traj():
    traj = fetch_test_file("example_traj.xtc")
    return traj
