import pytest
from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpansionSet
from asapdiscovery.data.state_expanders.tautomer_expander import TautomerExpander
from asapdiscovery.data.state_expanders.protomer_expander import ProtomerExpander
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander


@pytest.fixture(scope="session")
def wafarin_smi():
    return oe_smiles_roundtrip("CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O")


def test_expand_from_mol(wafarin_smi):
    l1 = Ligand.from_smiles(wafarin_smi, compound_name="test")
    stereo_expander = StereoExpander(stereo_expand_defined=True)
    ligands = stereo_expander.expand(ligands=[l1])
    assert len(ligands) == 2
    protomer_expander = ProtomerExpander()
    ligands = protomer_expander.expand(ligands=ligands)
    assert len(ligands) == 4


def test_expand_from_mol_collect_graph(wafarin_smi):
    l1 = Ligand.from_smiles(wafarin_smi, compound_name="test")
    stereo_expander = StereoExpander(stereo_expand_defined=True)
    ligands = stereo_expander.expand(ligands=[l1])
    assert len(ligands) == 2
    protomer_expander = ProtomerExpander()
    ligands = protomer_expander.expand(ligands=ligands)
    assert len(ligands) == 4
    state_expansion_set = StateExpansionSet.from_ligands(ligands)
    graph = state_expansion_set.to_networkx()
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4
