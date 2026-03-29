import pytest
from pydantic import ValidationError

from asapdiscovery.data.backend.openeye import load_openeye_pdb
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


def test_complex_from_oemol(complex_pdb):
    complex_mol = load_openeye_pdb(complex_pdb)

    c = Complex.from_oemol(
        complex_mol,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )

    assert c.target.target_name == "test"
    assert c.ligand.compound_name == "test"
    assert c.target.to_oemol().NumAtoms() == 4716
    assert c.ligand.to_oemol().NumAtoms() == 53


def test_complex_from_pdb(complex_pdb):
    c = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )

    assert c.target.target_name == "test"
    assert c.ligand.compound_name == "test"
    assert c.target.to_oemol().NumAtoms() == 4716
    assert c.ligand.to_oemol().NumAtoms() == 53


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
    c2 = Complex.from_dict(c1.model_dump())

    assert c1 == c2


def test_complex_json_roundtrip(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = Complex.from_json(c1.model_dump_json())

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
