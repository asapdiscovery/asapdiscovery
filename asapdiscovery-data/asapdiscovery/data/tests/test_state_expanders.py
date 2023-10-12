import pytest
from asapdiscovery.data.openeye import oe_smiles_roundtrip
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.protomer_expander import (
    EpikExpander,
    ProtomerExpander,
)

# from asapdiscovery.data.state_expanders.state_expander import (
#     StateExpansion,
#     StateExpansionSet,
# )
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.data.state_expanders.tautomer_expander import TautomerExpander
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
        pytest.param(
            "CC[C@@H](O1)CC[C@]12CCCO2", False, 1, id="All defined, don't expand"
        ),
        pytest.param("CC[C@@H](O1)CC[C@@]12CCCO2", True, 4, id="All defined expand"),
    ],
)
def test_expand_stereo(input_molecule, stereo_expand_defined, expected_states):
    l1 = Ligand.from_smiles(input_molecule, compound_name="test")
    # make sure we get a different molecule back
    expander = StereoExpander(stereo_expand_defined=stereo_expand_defined)
    ligands = expander.expand(ligands=[l1])
    for ligand in ligands:
        assert ligand.expansion_tag.parent_fixed_inchikey == l1.fixed_inchikey
        assert ligand.expansion_tag.provenance == expander.provenance()
    assert len(ligands) == expected_states


def test_expand_stereo_not_possible():
    """Make sure molecules for which there are no possible stereo expansions are passed through the stage and not tagged"""

    molecules = [
        Ligand.from_smiles("CCO", compound_name="ethanol"),
        Ligand.from_smiles("CC", compound_name="ethane"),
    ]

    expander = StereoExpander()
    expanded_ligands = expander.expand(ligands=molecules)

    assert len(expanded_ligands) == len(molecules)
    for lig in expanded_ligands:
        assert lig in molecules
        assert lig.expansion_tag is None


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


def test_epik_mocked_normal(monkeypatch):
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


def test_epik_mocked_no_expansions(monkeypatch):
    """Test running epik on a molecule with no expansions"""

    # a ligand which we know epik returns enumerated states
    ligand = Ligand.from_smiles("CC", compound_name="ethane")
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
        l1.set_SD_data({"r_epik_State_Penalty": -0.00})
        return [l1]

    monkeypatch.setattr(EpikExpander, "_extract_ligands", _return_expanded)

    expanded_ligands = epik_expander.expand(ligands=[ligand])
    assert len(expanded_ligands) == 1
    assert expanded_ligands[0].expansion_tag is None
    assert expanded_ligands[0] == ligand


def test_openeye_protomer_expander():
    """Test using the default openeye protomer expander."""

    ligand = Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="test")
    expander = ProtomerExpander()
    expanded_states = expander.expand(ligands=[ligand])
    assert len(expanded_states) == 2
    for expanded_ligand in expanded_states:
        # the parent ligand will not have a tag
        if expanded_ligand.fixed_inchikey == ligand.fixed_inchikey:
            assert expanded_ligand.expansion_tag is None
        else:
            # make sure the parent is set correctly on the new microstate
            assert (
                expanded_ligand.expansion_tag.parent_fixed_inchikey
                == ligand.fixed_inchikey
            )
            assert expanded_ligand.expansion_tag.provenance == expander.provenance()


def test_openeye_tautomer_expander():
    """Test using the default openeye tautomer expander."""

    ligand = Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="test")
    expander = TautomerExpander()
    expanded_ligands = expander.expand(ligands=[ligand])
    assert len(expanded_ligands) == 2
    for expanded_ligand in expanded_ligands:
        if expanded_ligand.fixed_inchikey == ligand.fixed_inchikey:
            assert expanded_ligand.expansion_tag is None
        else:
            assert (
                expanded_ligand.expansion_tag.parent_fixed_inchikey
                == ligand.fixed_inchikey
            )
            assert expanded_ligand.expansion_tag.provenance == expander.provenance()
