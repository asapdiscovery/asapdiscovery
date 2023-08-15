import pytest
from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpansion
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.data.testing.test_resources import fetch_test_file
from networkx.utils import graphs_equal


@pytest.fixture(scope="session")
def chalcogran_defined():
    # test file with two stereocenters defined
    smi_file = fetch_test_file("chalcogran_defined.smi")
    return smi_file


@pytest.fixture(scope="session")
def chalcogran_defined_smi(chalcogran_defined):
    return oe_smiles_roundtrip("CC[C@H](O1)CC[C@@]12CCCO2")


def test_expand_from_mol(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander()
    expansions = expander.expand(ligands=[l1])
    assert len(expansions) == 1
    assert expansions[0].parent == l1
    assert expansions[0].n_expanded_states == 1
    assert expansions[0].children[0].smiles == chalcogran_defined_smi


def test_expand_from_mol_tags(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander()
    expansions = expander.expand(ligands=[l1])
    assert l1.expansion_tag.is_parent
    children = StateExpansion.flatten_children(expansions)
    child = children[0]
    assert child.expansion_tag.is_child_of(l1.expansion_tag)
    assert l1.expansion_tag.is_parent_of(child.expansion_tag)


def test_expand_from_mol_tags_networkx(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander()
    expansions = expander.expand(ligands=[l1])
    children = StateExpansion.flatten_children(expansions)
    child = children[0]
    graph = StateExpansion.to_networkx(expansions)
    assert graph.has_edge(l1, child)

    graph2 = StateExpansion.ligands_to_networkx([l1, child])
    assert graph2.has_edge(l1, child)
    assert graph2.nodes == graph.nodes
    assert graph2.edges == graph.edges

    assert graphs_equal(graph, graph2)


def test_expand_from_mol_expand_defined(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    expansions = expander.expand(ligands=[l1])
    assert len(expansions) == 1
    assert expansions[0].parent == l1
    assert expansions[0].n_expanded_states == 4


def test_expand_from_mol_expand_defined_multi(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    expansions = expander.expand(ligands=[l1, l1])
    assert len(expansions) == 2
    assert expansions[0].parent == l1
    assert expansions[0].n_expanded_states == 4
    assert expansions[1].parent == l1
    assert expansions[1].n_expanded_states == 4


def test_expand_from_mol_expand_defined_multi_flatten(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    l2 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    expansions = expander.expand(ligands=[l1, l2])
    all_children = StateExpansion.flatten_children(expansions)
    # they recieve different expansion tags
    assert len(all_children) == 8
    assert len(set(all_children)) == 4
    all_parents = StateExpansion.flatten_parents(expansions)
    # they recieve different expansion tags
    assert len(all_parents) == 2
    assert len(set(all_parents)) == 1


def test_stereo_provenance(chalcogran_defined_smi):
    """Make sure the provenance of the state expander is correctly captured"""
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    expansion = expander.expand(ligands=[l1])[0]

    assert expansion.expander == expander.dict()
    assert "oechem" in expansion.provenance
    assert "omega" in expansion.provenance
