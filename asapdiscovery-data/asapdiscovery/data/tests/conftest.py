import pytest
from asapdiscovery.data.cdd_api import CDDAPI
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.services_config import CDDSettings
from asapdiscovery.data.testing.test_resources import fetch_test_file


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
def smiles():
    # smiles for the ligands in the first 4  test pdb files
    return [
        "Cc1ccncc1N(C)C(=O)Cc2cccc(c2)Cl",
        "CC(=O)N1CCN(CC1)c2ccc(cc2)OC",
        "c1cc(sc1)C(=O)NC(Cc2ccc(s2)N3CCOCC3)C=O",
        "c1cc[nH]c(=O)c1",
    ]


@pytest.fixture(scope="module")
def ligands(smiles):
    return [Ligand.from_smiles(s, compound_name="test") for s in smiles]


@pytest.fixture()
def mocked_cdd_api():
    """A cdd_api configured with dummy data which should have the requests mocked."""
    settings = CDDSettings(CDD_API_KEY="my-key", CDD_VAULT_NUMBER=1)
    return CDDAPI.from_settings(settings=settings)
