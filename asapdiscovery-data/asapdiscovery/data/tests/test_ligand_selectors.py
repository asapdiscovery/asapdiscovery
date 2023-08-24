import pytest
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.selectors.mcs_selector import MCSLigandSelector
from asapdiscovery.data.selectors.pairwise_selector import PairwiseLigandSelector
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


def test_pairwise_selector(ligands, complexes):
    selector = PairwiseLigandSelector()
    pairs = selector.select(ligands, complexes)
    assert len(pairs) == 40


def test_mcs_selector(ligands, complexes):
    selector = MCSLigandSelector()
    pairs = selector.select(ligands, complexes, n_draw=1)
    # should be 4 pairs
    assert len(pairs) == 4
    # as we matched against the exact smiles of the first 4 complex ligands, they should be in order
    assert pairs[0] == (ligands[0], complexes[0])
    assert pairs[1] == (ligands[1], complexes[1])
    assert pairs[2] == (ligands[2], complexes[2])
    assert pairs[3] == (ligands[3], complexes[3])


def test_mcs_selector(ligands, complexes):
    selector = MCSLigandSelector()
    pairs = selector.select(ligands, complexes, n_draw=2)
    # should be 8 pairs
    assert len(pairs) == 8
    assert pairs[0][1].ligand.smiles == "Cc1ccncc1N(C)C(=O)Cc2cccc(c2)Cl"  # exact match
    assert (
        pairs[1][1].ligand.smiles == "Cc1ccncc1NC(=O)Cc2cc(cc(c2)Cl)OC"
    )  # clearly related
