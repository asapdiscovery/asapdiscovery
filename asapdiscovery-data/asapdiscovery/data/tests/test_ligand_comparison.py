import pytest
from asapdiscovery.data.schema.ligand import ChemicalRelationship, Ligand


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
def acid_stereoisomer():
    return Ligand.from_smiles(
        compound_name="EDG-MED-5d232de5-4_acid_stereoisomer",
        smiles="CC(N1CC[C@H](C([NH2+]c2c3ccccc3cnc2)=O)c4cc(Cl)ccc41)=O",
    )


@pytest.fixture(scope="session")
def other():
    return Ligand.from_smiles(
        compound_name="other",
        smiles="CCCCCC",
    )


def test_get_2d_ligand(base_ligand, stereoisomer):
    assert base_ligand.flattened.is_chemically_equal(stereoisomer.flattened)
    assert not base_ligand.is_chemically_equal(stereoisomer)


def test_perceived_stereochemistry(base_ligand, stereoisomer, other):
    # Any molecule with a stereo center should have perceived stereo
    assert not other.has_perceived_stereo
    assert base_ligand.has_perceived_stereo
    assert stereoisomer.has_perceived_stereo
    assert base_ligand.flattened.has_perceived_stereo
    assert stereoisomer.flattened.has_perceived_stereo


def test_defined_stereochemistry(base_ligand, stereoisomer, other):
    # Molecules with stereo centers that are defined should have defined stereo
    assert base_ligand.has_defined_stereo
    assert stereoisomer.has_defined_stereo
    assert not base_ligand.flattened.has_defined_stereo
    assert not stereoisomer.flattened.has_defined_stereo
    assert not other.has_defined_stereo


def test_true_chemical_comparisons(
    base_ligand,
    different_name,
    stereoisomer,
    tautomer,
    acid,
    acid_stereoisomer,
):
    assert base_ligand == base_ligand

    # this is weird but bc `data_equal` ignores the name, this is true
    assert base_ligand == different_name

    assert base_ligand.is_chemically_equal(different_name)
    assert base_ligand.is_stereoisomer(stereoisomer)
    assert base_ligand.is_tautomer(tautomer)
    assert base_ligand.is_protonation_state_isomer(acid)

    assert acid.is_stereoisomer(acid_stereoisomer)
    assert stereoisomer.is_protonation_state_isomer(acid_stereoisomer)


def test_false_chemical_comparisons(
    base_ligand, different_name, stereoisomer, tautomer, acid, acid_stereoisomer, other
):
    for query in [stereoisomer, tautomer, acid]:
        assert not base_ligand.is_chemically_equal(query)
    for query in [different_name, acid, other]:
        assert not base_ligand.is_stereoisomer(query)
    for query in [different_name, stereoisomer, acid, other]:
        assert not base_ligand.is_tautomer(query)
    for query in [different_name, stereoisomer, tautomer, other]:
        assert not base_ligand.is_protonation_state_isomer(query)

    assert not stereoisomer.is_tautomer(tautomer)
    assert not stereoisomer.is_stereoisomer(tautomer)
    assert not stereoisomer.is_protonation_state_isomer(acid)

    assert not tautomer.is_tautomer(acid)
    assert not tautomer.is_stereoisomer(acid)
    assert not tautomer.is_protonation_state_isomer(acid)

    assert not tautomer.is_tautomer(acid_stereoisomer)
    assert not tautomer.is_stereoisomer(acid_stereoisomer)
    assert not tautomer.is_protonation_state_isomer(acid_stereoisomer)

    assert not acid.is_tautomer(acid_stereoisomer)


def test_base_chemical_relationships(
    base_ligand, different_name, stereoisomer, tautomer, acid, acid_stereoisomer, other
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
    assert (
        base_ligand.get_chemical_relationship(acid)
        == ChemicalRelationship.PROTONATION_STATE_ISOMER
    )

    assert base_ligand.get_chemical_relationship(other) == ChemicalRelationship.DISTINCT

    assert (
        acid.get_chemical_relationship(acid_stereoisomer)
        == ChemicalRelationship.STEREOISOMER
    )

    assert (
        stereoisomer.get_chemical_relationship(acid_stereoisomer)
        == ChemicalRelationship.PROTONATION_STATE_ISOMER
    )


def test_weird_chemical_relationships(stereoisomer, tautomer, acid, acid_stereoisomer):
    assert (
        stereoisomer.get_chemical_relationship(tautomer)
        == ChemicalRelationship.STEREOISOMER | ChemicalRelationship.TAUTOMER
    )
    assert (
        stereoisomer.get_chemical_relationship(acid)
        == ChemicalRelationship.STEREOISOMER
        | ChemicalRelationship.PROTONATION_STATE_ISOMER
    )
    assert (
        tautomer.get_chemical_relationship(acid)
        == ChemicalRelationship.TAUTOMER | ChemicalRelationship.PROTONATION_STATE_ISOMER
    )

    assert (
        tautomer.get_chemical_relationship(acid_stereoisomer)
        == ChemicalRelationship.TAUTOMER
        | ChemicalRelationship.STEREOISOMER
        | ChemicalRelationship.PROTONATION_STATE_ISOMER
    )


def test_using_chemical_relationship_flags(
    base_ligand, different_name, tautomer, stereoisomer, acid, acid_stereoisomer, other
):
    stereoisomerically_related = (
        ChemicalRelationship.IDENTICAL | ChemicalRelationship.STEREOISOMER
    )

    loosely_related = (
        ChemicalRelationship.IDENTICAL
        | ChemicalRelationship.STEREOISOMER
        | ChemicalRelationship.TAUTOMER
        | ChemicalRelationship.PROTONATION_STATE_ISOMER
    )

    assert (
        base_ligand.get_chemical_relationship(different_name)
        in stereoisomerically_related
    )
    assert (
        base_ligand.get_chemical_relationship(stereoisomer)
        in stereoisomerically_related
    )
    for query in [tautomer, acid, acid_stereoisomer, other]:
        assert (
            base_ligand.get_chemical_relationship(query)
            not in stereoisomerically_related
        )

    for query in [tautomer, acid, acid_stereoisomer]:
        assert base_ligand.get_chemical_relationship(query) in loosely_related
        assert stereoisomer.get_chemical_relationship(query) in loosely_related
        assert tautomer.get_chemical_relationship(query) in loosely_related
        assert acid.get_chemical_relationship(query) in loosely_related
