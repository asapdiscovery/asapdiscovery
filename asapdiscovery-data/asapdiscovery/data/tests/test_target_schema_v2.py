import pytest
from asapdiscovery.data.schema_v2.dynamic_properties import TargetType
from asapdiscovery.data.schema_v2.target import Target, TargetIdentifiers
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def moonshot_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")  # has a whole bunch of cruft in it
    return pdb


@pytest.fixture(scope="session")
def moonshot_pdb_processed():
    pdb = fetch_test_file(
        "Mpro-P2660_0A_bound_oe_processed.pdb"
    )  # already been processed with openeye
    return pdb


@pytest.fixture(scope="session")
def sars2_spruced_pdb():
    pdb = fetch_test_file("sars_spruced.pdb")
    return pdb


@pytest.mark.parametrize("ttype", ["sars2_mpro", "mers_mpro", "sars2_mac1"])
def test_targettype_init(ttype):
    tt = TargetType(ttype)
    assert tt.value == ttype


def test_targettype_init_bad_name():
    with pytest.raises(ValueError):
        _ = TargetType("bad_name")


@pytest.mark.parametrize("ttype", ["sars2_mpro", "mers_mpro", "sars2_mac1"])
def test_target_identifiers(ttype):
    ids = TargetIdentifiers(target_type=ttype, fragalysis_id="blah", pdb_code="blah")
    assert ids.target_type == TargetType(ttype)
    assert ids.fragalysis_id == "blah"
    assert ids.pdb_code == "blah"


@pytest.mark.parametrize("pdb_code", ["ABCD", None])
@pytest.mark.parametrize("fragalysis_id", ["Mpro-P2660", None])
@pytest.mark.parametrize("ttype", ["sars2_mpro", "mers_mpro", "sars2_mac1"])
@pytest.mark.parametrize("target_name", ["test_name"])
def test_target_dict_roundtrip(
    moonshot_pdb, target_name, ttype, fragalysis_id, pdb_code
):
    t1 = Target.from_pdb(
        moonshot_pdb,
        target_name,
        ids=TargetIdentifiers(
            target_type=ttype, fragalysis_id=fragalysis_id, pdb_code=pdb_code
        ),
    )
    t2 = Target.from_dict(t1.dict())
    assert t1 == t2


@pytest.mark.parametrize("pdb_code", ["ABCD", None])
@pytest.mark.parametrize("fragalysis_id", ["Mpro-P2660", None])
@pytest.mark.parametrize("ttype", ["sars2_mpro", "mers_mpro", "sars2_mac1"])
@pytest.mark.parametrize("target_name", ["test_name"])
def test_target_json_roundtrip(
    moonshot_pdb, target_name, ttype, fragalysis_id, pdb_code
):
    t1 = Target.from_pdb(
        moonshot_pdb,
        target_name,
        ids=TargetIdentifiers(
            target_type=ttype, fragalysis_id=fragalysis_id, pdb_code=pdb_code
        ),
    )
    t2 = Target.from_json(t1.json())
    assert t1 == t2


def test_target_data_equal(moonshot_pdb):
    t1 = Target.from_pdb(moonshot_pdb, "TargetTestName")
    t2 = Target.from_pdb(moonshot_pdb, "TargetTestName")
    # does the same thing as the __eq__ method
    assert t1.data_equal(t2)
    assert t1 == t2


def test_oemol_roundtrip(
    moonshot_pdb_processed,
):  # test that pre-processed pdb files can be read in and out consistently
    t1 = Target.from_pdb(moonshot_pdb_processed, "TargetTestName")
    mol = t1.to_oemol()
    t2 = Target.from_oemol(mol, "TargetTestName")
    assert t1 == t2


def test_oemol_roundtrip_sars2(
    sars2_spruced_pdb,
):  # test that a pdb file can be read in and out consistently via roundtrip through openeye
    t1 = Target.from_pdb(sars2_spruced_pdb, "TargetTestName")
    mol = t1.to_oemol()
    t2 = Target.from_oemol(mol, "TargetTestName")
    assert t1 == t2
