import pytest
from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.tautomer_expander import TautomerExpander
from asapdiscovery.data.state_expanders.protomer_expander import ProtomerExpander


@pytest.fixture(scope="session")
def wafarin_smi():
    return oe_smiles_roundtrip("CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O")


def test_expand_from_mol_tautomer(wafarin_smi):
    l1 = Ligand.from_smiles(wafarin_smi, compound_name="test")
    expander = TautomerExpander()
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == 2


def test_expand_from_mol_protomer(wafarin_smi):
    l1 = Ligand.from_smiles(wafarin_smi, compound_name="test")
    expander = ProtomerExpander()
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == 2
