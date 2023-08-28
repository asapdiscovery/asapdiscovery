import pytest
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair, DockingInputPair
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.selectors.pairwise_selector import PairwiseSelector
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


def test_pairwise_selector(ligands, complexes):
    selector = PairwiseSelector()
    pairs = selector.select(ligands, complexes)
    assert len(pairs) == 40


def test_pairwise_selector_prepped(ligands, prepped_complexes):
    selector = PairwiseSelector()
    pairs = selector.select(ligands, prepped_complexes)
    assert len(pairs) == 8


def test_mcs_selector(ligands, complexes):
    selector = MCSSelector()
    pairs = selector.select(ligands, complexes, n_select=1)
    # should be 4 pairs
    assert len(pairs) == 4
    # as we matched against the exact smiles of the first 4 complex ligands, they should be in order
    assert pairs[0] == CompoundStructurePair(ligand=ligands[0], complex=complexes[0])
    assert pairs[1] == CompoundStructurePair(ligand=ligands[1], complex=complexes[1])
    assert pairs[2] == CompoundStructurePair(ligand=ligands[2], complex=complexes[2])
    assert pairs[3] == CompoundStructurePair(ligand=ligands[3], complex=complexes[3])


def test_mcs_select_prepped(ligands, prepped_complexes):
    selector = MCSSelector()
    pairs = selector.select(ligands, prepped_complexes, n_select=1)
    # should be 4 pairs
    assert len(pairs) == 4
    assert pairs[0] == DockingInputPair(ligand=ligands[0], complex=prepped_complexes[0])
    assert pairs[1] == DockingInputPair(ligand=ligands[1], complex=prepped_complexes[1])
    assert pairs[2] == DockingInputPair(ligand=ligands[2], complex=prepped_complexes[1])
    assert pairs[3] == DockingInputPair(ligand=ligands[3], complex=prepped_complexes[0])


def test_mcs_selector_ndraw(ligands, complexes):
    selector = MCSSelector()
    pairs = selector.select(ligands, complexes, n_select=2)
    # should be 8 pairs
    assert len(pairs) == 8
    assert (
        pairs[0].complex.ligand.smiles == "Cc1ccncc1N(C)C(=O)Cc2cccc(c2)Cl"
    )  # exact match
    assert (
        pairs[1].complex.ligand.smiles == "Cc1ccncc1NC(=O)Cc2cc(cc(c2)Cl)OC"
    )  # clearly related
