import pytest
from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import (
    StateExpansion,
    StateExpansionSet,
)
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def chalcogran_defined():
    # test file with two stereocenters defined
    smi_file = fetch_test_file("chalcogran_defined.smi")
    return smi_file


@pytest.fixture(scope="session")
def chalcogran_defined_smi():
    return oe_smiles_roundtrip("CC[C@H](O1)CC[C@@]12CCCO2")


def test_expand_from_mol(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander()
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == 1
    child = ligands[0]
    assert child.expansion_tag.is_child_of(l1.expansion_tag)
    assert l1.expansion_tag.is_parent_of(child.expansion_tag)
    assert child.smiles == chalcogran_defined_smi


def test_expand_from_mol_collect(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander()
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == 1
    child = ligands[0]
    ses = StateExpansionSet.from_ligands([l1, child])
    assert ses.expansions[0] == StateExpansion(parent=l1, children=[child])


def test_expand_from_mol_expand_defined(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == 4


def test_expand_from_expand_defined_networkx(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    ligands = expander.expand(ligands=[l1])
    ses = StateExpansionSet.from_ligands(ligands)
    graph = ses.to_networkx()
    assert graph.has_edge(l1, ligands[0])


def test_expand_from_mol_expand_defined_multi(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    ligands = expander.expand(ligands=[l1, l1])
    assert len(ligands) == 4
    assert len(set(ligands)) == 4


def test_expand_from_mol_expand_defined_multi_non_unique(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    ligands = expander.expand(ligands=[l1, l1], unique=False)
    assert len(ligands) == 8
    assert len(set(ligands)) == 4


def test_stereo_provenance(chalcogran_defined_smi):
    """Make sure the provenance of the state expander is correctly captured"""
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    ligands = expander.expand(ligands=[l1])
    l0 = ligands[0]
    assert "expander" in l0.expansion_tag.provenance
    assert "oechem" in l0.expansion_tag.provenance
    assert "omega" in l0.expansion_tag.provenance
