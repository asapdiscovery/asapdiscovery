import pytest
from asapdiscovery.data.backend.openeye import oe_smiles_roundtrip
from asapdiscovery.data.operators.state_expanders.protomer_expander import (
    ProtomerExpander,
)
from asapdiscovery.data.operators.state_expanders.tautomer_expander import (
    TautomerExpander,
)
from asapdiscovery.data.schema.ligand import Ligand


@pytest.fixture(scope="session")
def wafarin_smi():
    return oe_smiles_roundtrip("CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O")


def test_expand_from_mol_tautomer(wafarin_smi):
    "Test the tautomer expander works as expected and returns the input molecule."
    l1 = Ligand.from_smiles(wafarin_smi, compound_name="test")
    expander = TautomerExpander()
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == 3
    assert l1 in ligands


def test_expand_from_mol_protomer(wafarin_smi):
    """Make sure the protomer expander works as expected and returns the input molecule"""
    l1 = Ligand.from_smiles(wafarin_smi, compound_name="test")
    expander = ProtomerExpander()
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == 3
    # make sure the input is retained
    assert l1 in ligands
