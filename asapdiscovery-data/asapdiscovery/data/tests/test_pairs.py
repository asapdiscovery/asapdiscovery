import pytest
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair, DockingInputPair
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


def test_compoundstructure_pair(complex_pdb):
    c = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    l = Ligand.from_smiles("c1cc[nH]c(=O)c1", compound_name="test")
    _ = CompoundStructurePair(complex=c, ligand=l)


def test_dockinginput_pair_from_compoundstructure_pair(complex_pdb):
    c = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    l = Ligand.from_smiles("c1cc[nH]c(=O)c1", compound_name="test")
    cs_pair = CompoundStructurePair(complex=c, ligand=l)
    _ = DockingInputPair.from_compound_structure_pair(cs_pair)


def test_dockinginput_pair(complex_pdb):
    c = PreppedComplex.from_complex(
        Complex.from_pdb(
            complex_pdb,
            target_kwargs={"target_name": "test"},
            ligand_kwargs={"compound_name": "test"},
        )
    )
    l = Ligand.from_smiles("c1cc[nH]c(=O)c1", compound_name="test")
    _ = DockingInputPair(complex=c, ligand=l)
