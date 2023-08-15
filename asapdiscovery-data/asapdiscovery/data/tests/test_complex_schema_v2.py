from pydantic import ValidationError
import pytest

from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.schema_v2.complex import Complex


@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb


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


def test_equals(complex_pdb):
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


def test_complex_from_pdb_needs_ids(complex_pdb):
    with pytest.raises(ValueError):
        c = Complex.from_pdb(complex_pdb, target_kwargs={"target_name": "test"})

    with pytest.raises(ValueError):
        c = Complex.from_pdb(complex_pdb, ligand_kwargs={"compound_name": "test"})

    with pytest.raises(ValueError):
        c = Complex.from_pdb(complex_pdb)


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
