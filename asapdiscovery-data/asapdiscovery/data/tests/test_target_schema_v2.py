import pytest
from asapdiscovery.data.openeye import load_openeye_design_unit
from asapdiscovery.data.schema_v2.target import PreppedTarget, Target, TargetIdentifiers
from asapdiscovery.data.testing.test_resources import fetch_test_file
from pydantic import ValidationError


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


@pytest.fixture(scope="session")
def oedu_file():
    oedu = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.oedu")
    return oedu


def test_target_from_pdb_at_least_one_id(moonshot_pdb):
    with pytest.raises(ValueError):
        # neither id is set
        Target.from_pdb(moonshot_pdb)


def test_target_from_pdb_at_least_one_target_id(moonshot_pdb):
    with pytest.raises(ValueError):
        # neither id is set
        Target.from_pdb(moonshot_pdb, ids=TargetIdentifiers())


@pytest.mark.parametrize(
    "ttype", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro", "SARS-CoV-2-Mac1"]
)
def test_target_identifiers(ttype):
    ids = TargetIdentifiers(target_type=ttype, fragalysis_id="blah", pdb_code="blah")
    assert ids.target_type.value == ttype
    assert ids.fragalysis_id == "blah"
    assert ids.pdb_code == "blah"


@pytest.mark.parametrize(
    "ttype", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro", "SARS-CoV-2-Mac1"]
)
def test_target_identifiers_json_file_roundtrip(ttype, tmp_path):
    ids = TargetIdentifiers(target_type=ttype, fragalysis_id="blah", pdb_code="blah")
    ids.to_json_file(tmp_path / "test.json")
    ids2 = TargetIdentifiers.from_json_file(tmp_path / "test.json")
    assert ids2.target_type.value == ttype
    assert ids2.fragalysis_id == "blah"
    assert ids2.pdb_code == "blah"


@pytest.mark.parametrize("pdb_code", ["ABCD", None])
@pytest.mark.parametrize("fragalysis_id", ["Mpro-P2660", None])
@pytest.mark.parametrize(
    "ttype", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro", "SARS-CoV-2-Mac1"]
)
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
@pytest.mark.parametrize(
    "ttype", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro", "SARS-CoV-2-Mac1"]
)
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


@pytest.mark.parametrize("pdb_code", ["ABCD", None])
@pytest.mark.parametrize("fragalysis_id", ["Mpro-P2660", None])
@pytest.mark.parametrize(
    "ttype", ["SARS-CoV-2-Mpro", "MERS-CoV-Mpro", "SARS-CoV-2-Mac1"]
)
@pytest.mark.parametrize("target_name", ["test_name"])
def test_target_json_file_roundtrip(
    moonshot_pdb, target_name, ttype, fragalysis_id, pdb_code, tmp_path
):
    t1 = Target.from_pdb(
        moonshot_pdb,
        target_name,
        ids=TargetIdentifiers(
            target_type=ttype, fragalysis_id=fragalysis_id, pdb_code=pdb_code
        ),
    )
    path = tmp_path / "test.json"
    t1.to_json_file(path)
    t2 = Target.from_json_file(path)
    assert t1 == t2


def test_target_data_equal(moonshot_pdb):
    t1 = Target.from_pdb(moonshot_pdb, "TargetTestName")
    t2 = Target.from_pdb(moonshot_pdb, "TargetTestName")
    # does the same thing as the __eq__ method
    assert t1.data_equal(t2)
    assert t1 == t2


def test_target_oemol_roundtrip(
    moonshot_pdb_processed,
):  # test that pre-processed pdb files can be read in and out consistently
    t1 = Target.from_pdb(moonshot_pdb_processed, "TargetTestName")
    mol = t1.to_oemol()
    t2 = Target.from_oemol(mol, "TargetTestName")
    assert t1 == t2


def test_target_oemol_roundtrip_sars2(
    sars2_spruced_pdb,
):  # test that a pdb file can be read in and out consistently via roundtrip through openeye
    t1 = Target.from_pdb(sars2_spruced_pdb, "TargetTestName")
    mol = t1.to_oemol()
    t2 = Target.from_oemol(mol, "TargetTestName")
    assert t1 == t2


# PreppedTarget tests


def test_preppedtarget_from_oedu_file(oedu_file):
    pt = PreppedTarget.from_oedu_file(oedu_file, "PreppedTargetTestName")
    oedu = pt.to_oedu()
    assert oedu.GetTitle() == "(AB) > LIG(A-403)"  # from one of the old files


def test_preppedtarget_from_oedu_file_at_least_one_id(oedu_file):
    with pytest.raises(ValidationError):
        # neither id is set
        PreppedTarget.from_oedu_file(oedu_file)


def test_preppedtarget_from_oedu_file_at_least_one_target_id(oedu_file):
    with pytest.raises(ValidationError):
        _ = PreppedTarget.from_oedu_file(oedu_file, ids=TargetIdentifiers())


def test_prepped_target_from_oedu_file_bad_file():
    with pytest.raises(FileNotFoundError):
        # neither id is set
        _ = PreppedTarget.from_oedu_file("bad_file", "PreppedTargetTestName")


def test_prepped_target_from_oedu(oedu_file):
    loaded_oedu = load_openeye_design_unit(oedu_file)
    loaded_oedu.SetTitle("PreppedTargetTestName")
    pt = PreppedTarget.from_oedu(loaded_oedu, "PreppedTargetTestName")
    oedu = pt.to_oedu()
    assert oedu.GetTitle() == "PreppedTargetTestName"


def test_prepped_target_from_oedu_file_roundtrip(oedu_file, tmp_path):
    pt = PreppedTarget.from_oedu_file(oedu_file, "PreppedTargetTestName")
    pt.to_oedu_file(tmp_path / "test.oedu")
    pt2 = PreppedTarget.from_oedu_file(tmp_path / "test.oedu", "PreppedTargetTestName")
    # these two compatisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)


def test_prepped_target_from_oedu_roundtrip(oedu_file):
    pt = PreppedTarget.from_oedu_file(oedu_file, "PreppedTargetTestName")
    du = pt.to_oedu()
    pt2 = PreppedTarget.from_oedu(du, "PreppedTargetTestName")
    # these two compatisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)


def test_prepped_target_json_roundtrip(oedu_file):
    pt = PreppedTarget.from_oedu_file(oedu_file, "PreppedTargetTestName")
    js = pt.json()
    pt2 = PreppedTarget.from_json(js)
    # these two compatisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)
    du = pt.to_oedu()
    assert du.GetTitle() == "(AB) > LIG(A-403)"


def test_prepped_target_json_fil_roundtrip(oedu_file, tmp_path):
    pt = PreppedTarget.from_oedu_file(oedu_file, "PreppedTargetTestName")
    path = tmp_path / "test.json"
    pt.to_json_file(path)
    pt2 = PreppedTarget.from_json_file(path)
    # these two compatisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)
    du = pt2.to_oedu()
    assert du.GetTitle() == "(AB) > LIG(A-403)"
