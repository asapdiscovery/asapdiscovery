import pytest

from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.protomer_expander import EpikExpander

# from asapdiscovery.data.state_expanders.state_expander import (
#     StateExpansion,
#     StateExpansionSet,
# )
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def chalcogran_defined():
    # test file with two stereocenters defined
    smi_file = fetch_test_file("chalcogran_defined.smi")
    return smi_file


@pytest.fixture(scope="session")
def chalcogran_defined_smi():
    return oe_smiles_roundtrip("CC[CH](O1)CC[C@@]12CCCO2")


@pytest.mark.parametrize(
    "input_molecule, stereo_expand_defined, expected_states",
    [
        pytest.param(
            "CC[CH](O1)CC[C]12CCCO2", False, 4, id="Mol undefined expand undefined"
        ),
        pytest.param("CC[CH](O1)CC[C]12CCCO2", True, 4, id="Mol undefined expand all"),
        pytest.param(
            "CC[CH](O1)CC[C@@]12CCCO2",
            False,
            2,
            id="Mol one defined expand undefined only",
        ),
        pytest.param(
            "CC[CH](O1)CC[C@@]12CCCO2", True, 4, id="Mol one defined expand all"
        ),
    ],
)
def test_expand_stereo(input_molecule, stereo_expand_defined, expected_states):
    l1 = Ligand.from_smiles(input_molecule, compound_name="test")
    # make sure we get a different molecule back
    expander = StereoExpander(stereo_expand_defined=stereo_expand_defined)
    ligands = expander.expand(ligands=[l1])
    assert len(ligands) == expected_states
    for ligand in ligands:
        assert ligand.expansion_tag.parent_fixed_inchikey == l1.fixed_inchikey
        assert ligand.expansion_tag.provenance == expander.provenance()


# def test_expand_from_expand_defined_networkx(chalcogran_defined_smi):
#     l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
#     expander = StereoExpander(stereo_expand_defined=True)
#     ligands = expander.expand(ligands=[l1])
#     ses = StateExpansionSet.from_ligands(ligands)
#     graph = ses.to_networkx()
#     assert graph.has_edge(l1, ligands[0])


def test_expand_from_mol_expand_defined_multi(chalcogran_defined_smi):
    l1 = Ligand.from_smiles(chalcogran_defined_smi, compound_name="test")
    expander = StereoExpander(stereo_expand_defined=True)
    ligands = expander.expand(ligands=[l1, l1])
    assert len(ligands) == 4
    assert len(set(ligands)) == 4


def test_epic_get_command(monkeypatch):
    """Test creating the epik command from an env variable."""
    monkeypatch.setenv("SCHRODINGER", "path/to/software")

    epik_expander = EpikExpander()
    cmd = epik_expander._create_cmd("my", "epik", "program")
    assert cmd == "path/to/software/my/epik/program"


def test_epik_mocked(monkeypatch):
    """Test running epik on a molecule with mocked results."""

    # a ligand which we know epik returns enumerated states
    ligand = Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="test")
    # mock the parent being set
    ligand.set_SD_data({"parent": ligand.fixed_inchikey})

    epik_expander = EpikExpander()

    # print(epik_expander.provenance())
    # mock the 4 function calls
    def _get_version(*args, **kwargs):
        return {"epik": 5}

    def _do_nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(EpikExpander, "_provenance", _get_version)
    monkeypatch.setattr(EpikExpander, "_prepare_ligands", _do_nothing)
    monkeypatch.setattr(EpikExpander, "_call_epik", _do_nothing)

    # mock the return of a ligand with labels
    def _return_expanded(*args, **kargs):
        "mock the returned expanded states"
        # set an epik score on the first ligand
        l1 = ligand
        l1.set_SD_data({"r_epik_State_Penalty": 0.1})
        # create another ligand with some large pentaly
        l2 = Ligand.from_smiles(
            "[H]C1=Nc2c(nc(nc2[O-])N([H])[H])N1[H]", compound_name="tautomer"
        )
        l2.set_SD_data({"parent": ligand.fixed_inchikey})
        l2.set_SD_data({"r_epik_State_Penalty": 2})
        return [l1, l2]

    monkeypatch.setattr(EpikExpander, "_extract_ligands", _return_expanded)

    expanded_ligands = epik_expander.expand(ligands=[ligand])
    assert len(expanded_ligands) == 2
    for lig in expanded_ligands:
        assert lig.expansion_tag.parent_fixed_inchikey == ligand.fixed_inchikey
