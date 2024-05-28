import pytest
from asapdiscovery.data.operators.symmetry_expander import SymmetryExpander
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture()
def compl():
    c = Complex.from_pdb(
        fetch_test_file("Mpro-P2660_0A_bound.pdb"),
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    return [c]


@pytest.mark.parametrize("use_dask", [True, False])
def test_expander(compl, use_dask):
    se = SymmetryExpander()
    exp = se.expand(compl)
    assert exp
    exp[0].to_pdb("tst.pdb")
