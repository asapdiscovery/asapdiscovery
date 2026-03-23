import pytest

from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


def test_compoundstructure_pair(complex_pdb):
    cmplx = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    lig = Ligand.from_smiles("c1cc[nH]c(=O)c1", compound_name="test")
    _ = CompoundStructurePair(complex=cmplx, ligand=lig)
