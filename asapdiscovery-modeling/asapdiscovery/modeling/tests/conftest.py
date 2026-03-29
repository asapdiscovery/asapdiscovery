import pytest

from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.schema import PreppedComplex


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


@pytest.fixture(scope="session")
def complex_oedu():
    oedu = fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu")
    return oedu


@pytest.fixture(scope="session")
def oedu_file():
    oedu = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.oedu")
    return oedu


@pytest.fixture(scope="session")
def all_mpro_fns():
    return [
        "aligned/Mpro-x11041_0A/Mpro-x11041_0A_bound.pdb",
        "aligned/Mpro-x1425_0A/Mpro-x1425_0A_bound.pdb",
        "aligned/Mpro-x11894_0A/Mpro-x11894_0A_bound.pdb",
        "aligned/Mpro-x1002_0A/Mpro-x1002_0A_bound.pdb",
        "aligned/Mpro-x10155_0A/Mpro-x10155_0A_bound.pdb",
        "aligned/Mpro-x0354_0A/Mpro-x0354_0A_bound.pdb",
        "aligned/Mpro-x11271_0A/Mpro-x11271_0A_bound.pdb",
        "aligned/Mpro-x1101_1A/Mpro-x1101_1A_bound.pdb",
        "aligned/Mpro-x1187_0A/Mpro-x1187_0A_bound.pdb",
        "aligned/Mpro-x10338_0A/Mpro-x10338_0A_bound.pdb",
    ]


@pytest.fixture(scope="session")
def complexes(all_mpro_fns):
    all_pdbs = [fetch_test_file(f"frag_factory_test/{fn}") for fn in all_mpro_fns]
    return [
        Complex.from_pdb(
            struct,
            target_kwargs={"target_name": "test"},
            ligand_kwargs={"compound_name": "test"},
        )
        for struct in all_pdbs
    ]


@pytest.fixture(scope="session")
def prepped_complexes(complexes):
    # kinda expensive to make, so let's just do the first 2
    return [PreppedComplex.from_complex(c) for c in complexes[:2]]


@pytest.fixture(scope="module")
def ligands_from_complexes(complexes):
    # get ligands from 3d structure to ensure the added hydrogens make sense, using top 4 to match the smiles
    return [c.ligand for c in complexes[0:4]]
