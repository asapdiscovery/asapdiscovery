import pytest

from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.schema.target import Target, TargetIdentifiers
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
        target_name=target_name,
        ids=TargetIdentifiers(
            target_type=ttype, fragalysis_id=fragalysis_id, pdb_code=pdb_code
        ),
    )
    t2 = Target.from_dict(t1.model_dump())
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
        target_name=target_name,
        ids=TargetIdentifiers(
            target_type=ttype, fragalysis_id=fragalysis_id, pdb_code=pdb_code
        ),
    )
    t2 = Target.from_json(t1.model_dump_json())
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
        target_name=target_name,
        ids=TargetIdentifiers(
            target_type=ttype, fragalysis_id=fragalysis_id, pdb_code=pdb_code
        ),
    )
    path = tmp_path / "test.json"
    t1.to_json_file(path)
    t2 = Target.from_json_file(path)
    assert t1 == t2


def test_target_data_equal(moonshot_pdb):
    t1 = Target.from_pdb(moonshot_pdb, target_name="TargetTestName")
    t2 = Target.from_pdb(moonshot_pdb, target_name="TargetTestName")
    # does the same thing as the __eq__ method
    assert t1.data_equal(t2)
    assert t1 == t2


def test_target_oemol_roundtrip(
    moonshot_pdb_processed,
):  # test that pre-processed pdb files can be read in and out consistently
    t1 = Target.from_pdb(moonshot_pdb_processed, target_name="TargetTestName")
    mol = t1.to_oemol()
    t2 = Target.from_oemol(mol, target_name="TargetTestName")
    assert t1 == t2


def test_target_oemol_roundtrip_sars2(
    sars2_spruced_pdb,
):  # test that a pdb file can be read in and out consistently via roundtrip through openeye
    t1 = Target.from_pdb(sars2_spruced_pdb, target_name="TargetTestName")
    mol = t1.to_oemol()
    t2 = Target.from_oemol(mol, target_name="TargetTestName")
    assert t1 == t2


def test_target_moonshot_pdb_processed_no_ligand(moonshot_pdb):
    t1 = Target.from_pdb(moonshot_pdb, target_name="TargetTestName")
    mol = t1.to_oemol()
    opts = oechem.OESplitMolComplexOptions()
    lig = oechem.OEGraphMol()
    prot = oechem.OEGraphMol()
    wat = oechem.OEGraphMol()
    other = oechem.OEGraphMol()

    oechem.OESplitMolComplex(lig, prot, wat, other, mol, opts)
    assert lig.NumAtoms() == 0
    assert prot.NumAtoms() != 0
    assert wat.NumAtoms() == 0
