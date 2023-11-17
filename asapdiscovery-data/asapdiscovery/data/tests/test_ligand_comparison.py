import pytest
from asapdiscovery.data.schema_v2.ligand import ChemicalRelationship, Ligand


@pytest.fixture(scope="session")
def base_ligand():
    return Ligand.from_smiles(
        compound_name="EDG-MED-5d232de5-3",
        smiles="CC(=O)N1CC[C@H](c2c1ccc(c2)Cl)C(=O)Nc3cncc4c3cccc4",
    )


@pytest.fixture(scope="session")
def different_name():
    return Ligand.from_smiles(
        compound_name="EDG-MED-5d232de5-3_with_a_different_name",
        smiles="CC(=O)N1CC[C@H](c2c1ccc(c2)Cl)C(=O)Nc3cncc4c3cccc4",
    )


@pytest.fixture(scope="session")
def stereoisomer():
    return Ligand.from_smiles(
        compound_name="EDG-MED-5d232de5-4",
        smiles="CC(=O)N1CC[C@@H](c2c1ccc(c2)Cl)C(=O)Nc3cncc4c3cccc4",
    )


@pytest.fixture(scope="session")
def tautomer():
    return Ligand.from_smiles(
        compound_name="EDG-MED-5d232de5-3_tautomer",
        smiles=r"CC(N1CC[C@@H](/C(O)=N\c2c3ccccc3cnc2)c4cc(Cl)ccc41)=O",
    )


@pytest.fixture(scope="session")
def acid():
    return Ligand.from_smiles(
        compound_name="EDG-MED-5d232de5-4_acid",
        smiles="CC(N1CC[C@@H](C([NH2+]c2c3ccccc3cnc2)=O)c4cc(Cl)ccc41)=O",
    )


@pytest.fixture(scope="session")
def other():
    return Ligand.from_smiles(
        compound_name="other",
        smiles="CCCCCC",
    )


def test_chemical_relationships(
    base_ligand, different_name, stereoisomer, tautomer, acid, other
):
    assert (
        base_ligand.get_chemical_relationship(different_name)
        == ChemicalRelationship.IDENTICAL
    )
    assert (
        base_ligand.get_chemical_relationship(stereoisomer)
        == ChemicalRelationship.STEREOISOMER
    )
    assert (
        base_ligand.get_chemical_relationship(tautomer) == ChemicalRelationship.TAUTOMER
    )
    assert base_ligand.get_chemical_relationship(acid) == ChemicalRelationship.TAUTOMER

    assert (
        stereoisomer.get_chemical_relationship(tautomer)
        == ChemicalRelationship.STEREOISOMER | ChemicalRelationship.TAUTOMER
    )
    assert (
        stereoisomer.get_chemical_relationship(acid)
        == ChemicalRelationship.STEREOISOMER | ChemicalRelationship.TAUTOMER
    )
    assert base_ligand.get_chemical_relationship(other) == ChemicalRelationship.DISTINCT


def test_using_chemical_relationship_flags(
    base_ligand, different_name, stereoisomer, other
):
    stereoisomerically_related = (
        ChemicalRelationship.IDENTICAL | ChemicalRelationship.STEREOISOMER
    )

    assert (
        base_ligand.get_chemical_relationship(different_name)
        in stereoisomerically_related
    )
    assert (
        base_ligand.get_chemical_relationship(stereoisomer)
        in stereoisomerically_related
    )
    assert (
        base_ligand.get_chemical_relationship(other) in stereoisomerically_related
    ) is False
