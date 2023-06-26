import pytest
from asapdiscovery.data.schema_v2.target import Target, TargetIdentifiers
from asapdiscovery.data.schema_v2.dynamic_properties import TargetType
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def moonshot_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


@pytest.fixture(scope="session")
def moonshot_pdb_contents(moonshot_pdb):
    with open(moonshot_pdb, "r") as f:
        return f.read()


@pytest.mark.parametrize("ttype", ["sars2", "mers", "mac1"])
def test_targettype_init(ttype):
    tt = TargetType(ttype)
    assert tt.value == ttype


def test_targettype_init_bad_name():
    with pytest.raises(ValueError):
        tt = TargetType("bad_name")


@pytest.mark.parametrize("ttype", ["sars2", "mers", "mac1"])
def test_target_identifiers(ttype):
    ids = TargetIdentifiers(target_type=ttype, fragalysis_id="blah", pdb_code="blah")
    print(ids)
    assert ids.target_type == TargetType(ttype)
    assert ids.fragalysis_id == "blah"
    assert ids.pdb_code == "blah"


@pytest.mark.parametrize("pdb_code", ["ABCD", None])
@pytest.mark.parametrize("fragalysis_id", ["Mpro-P2660", None])
@pytest.mark.parametrize("ttype", ["sars2", "mers", "mac1"])
@pytest.mark.parametrize("target_name", ["test_name", None])
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
@pytest.mark.parametrize("ttype", ["sars2", "mers", "mac1"])
@pytest.mark.parametrize("target_name", ["test_name", None])
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
