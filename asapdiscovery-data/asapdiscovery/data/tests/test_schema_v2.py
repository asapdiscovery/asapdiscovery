import os
import pytest

from asapdiscovery.data.schema_v2 import Ligand, LigandIdentifiers
from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.data.openeye import load_openeye_sdf
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def smiles():
    return "CCCCCCC"


@pytest.fixture(scope="session")
def moonshot_sdf():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    return sdf


def test_ligand_from_smiles(smiles):
    lig = Ligand.from_smiles(smiles)
    assert lig.smiles == smiles


def test_ligand_from_sdf(moonshot_sdf):
    lig = Ligand.from_sdf(moonshot_sdf)
    assert (
        lig.smiles == "c1ccc2c(c1)c(cc(=O)[nH]2)C(=O)NCCOc3cc(cc(c3)Cl)O[C@H]4CC(=O)N4"
    )


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("postera_vc_id", ["test_postera_vc_id", None])
@pytest.mark.parametrize("compound_name", ["test_name", None])
def test_ligand_dict_roundtrip(
    smiles, compound_name, postera_vc_id, moonshot_compound_id, exp_data
):
    l1 = Ligand.from_smiles(
        smiles,
        ids=LigandIdentifiers(
            postera_vc_id=postera_vc_id, moonshot_compound_id=moonshot_compound_id
        ),
        experimental_data=exp_data,
    )
    l2 = Ligand.from_dict(l1.dict())
    assert l1 == l2


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("postera_vc_id", ["test_postera_vc_id", None])
@pytest.mark.parametrize("compound_name", ["test_name", None])
def test_ligand_json_roundtrip(
    smiles, compound_name, postera_vc_id, moonshot_compound_id, exp_data
):
    l1 = Ligand.from_smiles(
        smiles,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            postera_vc_id=postera_vc_id, moonshot_compound_id=moonshot_compound_id
        ),
        experimental_data=exp_data,
    )
    l2 = Ligand.from_json(l1.json())
    assert l1 == l2


def test_ligand_sdf_rountrip(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf)
    l1.to_sdf("test.sdf")
    l2 = Ligand.from_sdf("test.sdf")
    assert l1 == l2
    os.unlink("test.sdf")


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("postera_vc_id", ["test_postera_vc_id", None])
@pytest.mark.parametrize("compound_name", ["test_name", None])
def test_ligand_sdf_rountrip_data_only(
    moonshot_sdf, compound_name, postera_vc_id, moonshot_compound_id, exp_data
):
    l1 = Ligand.from_sdf(
        moonshot_sdf,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            postera_vc_id=postera_vc_id, moonshot_compound_id=moonshot_compound_id
        ),
        experimental_data=exp_data,
    )
    l1.to_sdf("test.sdf")
    l2 = Ligand.from_sdf("test.sdf")
    assert l1.data_equal(l2)
    os.unlink("test.sdf")


def test_ligand_oemol_rountrip(moonshot_sdf):
    mol = load_openeye_sdf(str(moonshot_sdf))
    l1 = Ligand.from_oemol(mol)
    mol_res = l1.to_oemol()
    l2 = Ligand.from_oemol(mol_res)
    assert l2 == l1


def test_ligand_oemol_rountrip_data_only(moonshot_sdf):
    mol = load_openeye_sdf(str(moonshot_sdf))
    l1 = Ligand.from_oemol(mol, compound_name="blahblah")
    mol_res = l1.to_oemol()
    l2 = Ligand.from_oemol(mol_res)
    assert l1.data_equal(l2)
