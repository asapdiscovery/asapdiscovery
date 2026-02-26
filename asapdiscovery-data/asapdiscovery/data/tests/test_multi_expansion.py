import pytest
from asapdiscovery.data.backend.openeye import oe_smiles_roundtrip
from asapdiscovery.data.operators.state_expanders.protomer_expander import (
    ProtomerExpander,
)
from asapdiscovery.data.operators.state_expanders.state_expander import (
    StateExpansionSet,
)
from asapdiscovery.data.operators.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.data.schema.ligand import Ligand


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
    # we should have two protomers for each input molecule and the two inputs
    assert len(ligands) == 6


def test_expand_from_mol_collect_graph(wafarin_smi):
    l1 = Ligand.from_smiles(wafarin_smi, compound_name="warfarin")
    stereo_expander = StereoExpander(stereo_expand_defined=True)
    expanded_ligands = stereo_expander.expand(ligands=[l1])
    assert len(expanded_ligands) == 2
    # the stereo expander does not keep an undefined input so add it back
    expanded_ligands.append(l1)
    state_expansion_set = StateExpansionSet.from_ligands(expanded_ligands)
    # make sure the expansion is correctly grouped
    assert len(state_expansion_set.get_stereo_expansions()) == 1
    assert len(state_expansion_set.get_charge_expansions()) == 0

    protomer_expander = ProtomerExpander()
    # remove the l1 ligand
    expanded_ligands.remove(l1)
    ligands = protomer_expander.expand(ligands=expanded_ligands)
    assert len(ligands) == 6

    # group the ligands we should have two expansions one for each stereo isomer of the input molecule
    state_expansion_set = StateExpansionSet.from_ligands(ligands)
    assert len(state_expansion_set.get_stereo_expansions()) == 0
    assert len(state_expansion_set.get_charge_expansions()) == 2
