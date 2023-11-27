import pytest
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.testing.test_resources import fetch_test_file
from pydantic import ValidationError


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


@pytest.fixture(scope="session")
def complex_oedu():
    oedu = fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu")
    return oedu


def test_complex_from_pdb(complex_pdb):
    c = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )

    assert c.target.target_name == "test"
    assert c.ligand.compound_name == "test"
    assert c.target.to_oemol().NumAtoms() == 4716
    assert c.ligand.to_oemol().NumAtoms() == 33


def test_equal(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )

    assert c1 == c2


def test_data_equal(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )

    assert c1.data_equal(c2)
    assert c2.data_equal(c1)


def test_complex_from_pdb_needs_ids(complex_pdb):
    """Make sure an error is raised if we do not supply ligand and receptor ids"""
    with pytest.raises(ValidationError):
        Complex.from_pdb(complex_pdb)


def test_complex_dict_roundtrip(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = Complex.from_dict(c1.dict())

    assert c1 == c2


def test_complex_json_roundtrip(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = Complex.from_json(c1.json())

    assert c1 == c2


def test_complex_json_file_roundtrip(complex_pdb, tmp_path):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    path = tmp_path / "test.json"
    c1.to_json_file(path)
    c2 = Complex.from_json_file(path)

    assert c1 == c2


def test_prepped_complex_from_complex(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = PreppedComplex.from_complex(c1, prep_kwargs={})
    du = c2.target.to_oedu()
    assert du.HasReceptor()
    assert du.HasLigand()
    assert c2.target.target_name == "test"
    assert c2.ligand.compound_name == "test"


def test_prepped_complex_from_oedu_file(complex_oedu):
    c = PreppedComplex.from_oedu_file(
        complex_oedu,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    assert c.target.target_name == "test"
    assert c.ligand.compound_name == "test"
