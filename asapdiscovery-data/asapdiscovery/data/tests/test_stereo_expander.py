import pytest
from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpansion
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.data.testing.test_resources import fetch_test_file


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
    expander = StereoExpander(input_ligands=[l1])
    expansions = expander.expand()
    assert len(expansions) == 1
    assert expansions[0].parent == l1
    assert expansions[0].n_expanded_states == 1
    assert expansions[0].children[0].smiles == chalcogran_defined_smi


def test_expand_from_mol_expand_defined(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(input_ligands=[l1], stereo_expand_defined=True)
    expansions = expander.expand()
    assert len(expansions) == 1
    print(expansions)
    assert expansions[0].parent == l1
    assert expansions[0].n_expanded_states == 4


def test_expand_from_mol_expand_defined_multi(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(input_ligands=[l1, l1], stereo_expand_defined=True)
    expansions = expander.expand()
    assert len(expansions) == 2
    assert expansions[0].parent == l1
    assert expansions[0].n_expanded_states == 4
    assert expansions[1].parent == l1
    assert expansions[1].n_expanded_states == 4


def test_expand_from_mol_expand_defined_multi_flatten(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    l2 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(input_ligands=[l1, l2], stereo_expand_defined=True)
    expansions = expander.expand()
    all_children = StateExpansion.flatten_children(expansions)
    assert len(all_children) == 8
    assert len(set(all_children)) == 4
    all_parents = StateExpansion.flatten_parents(expansions)
    assert len(all_parents) == 2
    assert len(set(all_parents)) == 1
