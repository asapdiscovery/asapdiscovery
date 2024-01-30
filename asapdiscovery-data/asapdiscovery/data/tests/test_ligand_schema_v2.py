from uuid import uuid4

import pytest
from asapdiscovery.data.openeye import get_SD_data, load_openeye_sdf, set_SD_data
from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.identifiers import LigandIdentifiers, LigandProvenance
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def smiles():
    return "CCCCCCC"


@pytest.fixture(scope="session")
def moonshot_sdf():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    return sdf


def test_ligand_from_smiles(smiles):
    lig = Ligand.from_smiles(smiles, compound_name="test_name")
    assert lig.smiles == smiles


def test_from_smiles_ids_made(smiles):
    """Make sure the ligand provenance is automatically generated."""
    lig = Ligand.from_smiles(smiles, compound_name="test_name")
    assert lig.provenance.isomeric_smiles == smiles


def test_from_oemol_sd_tags_left(moonshot_sdf):
    """Make sure any sd tags on an oemol are not lost when building a ligand."""
    mol = load_openeye_sdf(str(moonshot_sdf))
    sd_data = {"compound_name": "moonshot-mol", "energy": "1"}
    set_SD_data(mol, sd_data)
    # create a ligand keeping the original sd safe
    _ = Ligand.from_oemol(mol)
    assert get_SD_data(mol) == sd_data


def test_ligand_from_smiles_hashable(smiles):
    lig1 = Ligand.from_smiles(smiles, compound_name="test_name")
    lig2 = Ligand.from_smiles(smiles, compound_name="test_name")
    lig3 = Ligand.from_smiles(smiles, compound_name="test_name")

    assert len({lig1, lig2, lig3}) == 1


def test_ligand_from_smiles_id(smiles):
    lig = Ligand.from_smiles(
        smiles, ids=LigandIdentifiers(moonshot_compound_id="test_id")
    )
    assert lig.smiles == smiles


def test_ligand_ids_round_trip(smiles, tmpdir):
    """Make sure ligand ids can survive a round trip to sdf"""
    with tmpdir.as_cwd():
        lig = Ligand.from_smiles(
            smiles, ids=LigandIdentifiers(moonshot_compound_id="test_id")
        )
        assert lig.ids is not None
        lig.to_sdf("test.sdf")

        lig2 = Ligand.from_sdf(sdf_file="test.sdf")
        assert lig2.ids is not None
        assert lig2.ids == lig.ids


def test_ligand_from_smiles_at_least_one_id(smiles):
    with pytest.raises(ValueError):
        # neither id is set
        Ligand.from_smiles(smiles)


def test_ligand_from_smiles_at_least_one_ligand_id(smiles):
    with pytest.raises(ValueError):
        # LigandIdentifiers is set but empty
        Ligand.from_smiles(smiles, ids=LigandIdentifiers())


def test_ligand_ids_json_roundtrip():
    ids = LigandIdentifiers(
        manifold_api_id=uuid4(),
        manifold_vc_id="ASAP-VC-1234",
        moonshot_compound_id="test_moonshot_compound_id",
        compchem_id=uuid4(),
    )
    ids2 = LigandIdentifiers.from_json(ids.json())
    assert ids == ids2


def test_ligand_ids_json_file_roundtrip(tmp_path):
    ids = LigandIdentifiers(
        manifold_api_id=uuid4(),
        manifold_vc_id="ASAP-VC-1234",
        moonshot_compound_id="test_moonshot_compound_id",
        compchem_id=uuid4(),
    )
    ids.to_json_file(tmp_path / "test.json")
    ids2 = LigandIdentifiers.from_json_file(tmp_path / "test.json")
    assert ids == ids2


def test_ligand_from_sdf(moonshot_sdf):
    lig = Ligand.from_sdf(moonshot_sdf, compound_name="test_name")
    assert (
        lig.smiles == "c1ccc2c(c1)c(cc(=O)[nH]2)C(=O)NCCOc3cc(cc(c3)Cl)O[C@H]4CC(=O)N4"
    )
    assert lig.compound_name == "test_name"


def test_ligand_from_sdf_title_used(moonshot_sdf):
    # make sure the ligand title is used as the compound ID if not set
    # important test this due to complicated skip and validation logic
    lig = Ligand.from_sdf(moonshot_sdf)
    assert (
        lig.smiles == "c1ccc2c(c1)c(cc(=O)[nH]2)C(=O)NCCOc3cc(cc(c3)Cl)O[C@H]4CC(=O)N4"
    )
    assert lig.compound_name == "Mpro-P0008_0A_ERI-UCB-ce40166b-17"


def test_inchi(smiles):
    lig = Ligand.from_smiles(smiles, compound_name="test_name")
    assert lig.inchi == "InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3"


def test_inchi_key(smiles):
    lig = Ligand.from_smiles(smiles, compound_name="test_name")
    assert lig.inchikey == "IMNFDUFMRHMDMM-UHFFFAOYSA-N"


def test_fixed_inchi():
    "Make sure a tautomer specific inchi is made when requested."
    lig = Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="test")
    assert (
        lig.fixed_inchi
        == "InChI=1/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)/f/h7,10H,6H2"
    )
    assert lig.fixed_inchi != lig.inchi


def test_fixed_inchikey():
    "Make sure a tautomer specific inchikey is made when requested."
    lig = Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="test")
    assert lig.fixed_inchikey == "UYTPUPDQBNUYGX-CQCWYMDMNA-N"
    assert lig.inchikey != lig.fixed_inchikey


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name"])
def test_ligand_dict_roundtrip(
    smiles,
    compound_name,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data,
):
    l1 = Ligand.from_smiles(
        smiles,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
        ),
        experimental_data=exp_data,
    )
    l2 = Ligand.from_dict(l1.dict())
    assert l1 == l2


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name"])
def test_ligand_json_roundtrip(
    smiles,
    compound_name,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data,
):
    l1 = Ligand.from_smiles(
        smiles,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
        ),
        experimental_data=exp_data,
    )
    l2 = Ligand.from_json(l1.json())
    assert l1 == l2


def test_ligand_sdf_roundtrip(moonshot_sdf, tmp_path):
    l1 = Ligand.from_sdf(moonshot_sdf, compound_name="test_name")
    l1.to_sdf(tmp_path / "test.sdf")
    l2 = Ligand.from_sdf(tmp_path / "test.sdf")
    assert l1 == l2


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compchem_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name"])
def test_ligand_sdf_roundtrip_data_only(
    moonshot_sdf,
    compound_name,
    compchem_id,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data,
    tmp_path,
):
    l1 = Ligand.from_sdf(
        moonshot_sdf,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
            compchem_id=compchem_id,
        ),
        experimental_data=exp_data,
    )
    l1.to_sdf(tmp_path / "test.sdf")
    l2 = Ligand.from_sdf(tmp_path / "test.sdf")
    # checks the same thing l1.data == l2.data
    assert l1.data_equal(l2)
    assert l1 == l2
    # checks every field
    assert l1.full_equal(l2)


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compchem_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name"])
def test_ligand_json_file_roundtrip(
    moonshot_sdf,
    compound_name,
    compchem_id,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data,
    tmp_path,
):
    l1 = Ligand.from_sdf(
        moonshot_sdf,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
            compchem_id=compchem_id,
        ),
        experimental_data=exp_data,
    )
    l1.to_json_file(tmp_path / "test.json")
    l2 = Ligand.from_json_file(tmp_path / "test.json")
    assert l1 == l2


def test_ligand_oemol_roundtrip(moonshot_sdf):
    mol = load_openeye_sdf(str(moonshot_sdf))
    l1 = Ligand.from_oemol(mol, compound_name="blahblah")
    mol_res = l1.to_oemol()
    l2 = Ligand.from_oemol(mol_res, compound_name="blahblah")
    assert l2 == l1
    # check all internal fields as well
    assert l2.dict() == l1.dict()


def test_ligand_oemol_roundtrip_data_only(moonshot_sdf):
    mol = load_openeye_sdf(str(moonshot_sdf))
    l1 = Ligand.from_oemol(mol, compound_name="blahblah")
    mol_res = l1.to_oemol()
    l2 = Ligand.from_oemol(mol_res, compound_name="blahblah")
    assert l1.data_equal(l2)


def test_get_set_sd_data(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf, compound_name="blahblah")
    data = {"test_key": "test_value", "test_key2": "test_value2", "test_key3": "3"}
    l1.set_SD_data(data)
    data_pulled = l1.get_SD_data()
    assert data_pulled == data


def test_print_sd_data(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf, compound_name="blahblah")
    data = {"test_key": "test_value", "test_key2": "test_value2", "test_key3": "3"}
    l1.set_SD_data(data)
    l1.print_SD_data()


def test_clear_sd_data(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf, compound_name="blahblah")
    data = {"test_key": "test_value", "test_key2": "test_value2", "test_key3": "3"}
    l1.set_SD_data(data)
    l1.clear_SD_data()
    assert l1.get_SD_data() == {}


def test_clear_sd_data_reserved_fails(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf, compound_name="blahblah")
    data = {"experimental_data": "blahblah"}
    with pytest.raises(ValueError):
        l1.set_SD_data(data)


@pytest.mark.parametrize("tags", [{"test_key": "test_value"}, {}])
@pytest.mark.parametrize("exp_data_vals", [{"pIC50": 5.0}, {}])
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compchem_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name"])
def test_ligand_sdf_roundtrip_SD(
    moonshot_sdf,
    compound_name,
    compchem_id,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data_vals,
    tags,
    tmp_path,
):
    exp_data = ExperimentalCompoundData(
        compound_id="blah", smiles="CCCC", experimental_data=exp_data_vals
    )
    l1 = Ligand.from_sdf(
        moonshot_sdf,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
            compchem_id=compchem_id,
        ),
        experimental_data=exp_data,
        tags=tags,
    )
    # serialize with SD data
    l1.to_sdf(tmp_path / "test_with_attrs.sdf")
    # read with SD data
    l2 = Ligand.from_sdf(tmp_path / "test_with_attrs.sdf")
    assert l1 == l2


def test_to_rdkit(smiles):
    """Make sure we can convert to an rdkit molecule without losing any SD tags."""

    molecule = Ligand.from_smiles(smiles=smiles, compound_name="testing")
    rdkit_mol = molecule.to_rdkit()
    props = rdkit_mol.GetPropsAsDict()
    # we only check the none default properties as these are what are saved
    assert molecule.compound_name == props["compound_name"]
    assert molecule.provenance == LigandProvenance.parse_raw(props["provenance"])
    assert molecule.data_format.value == props["data_format"]