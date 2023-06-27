from uuid import UUID, uuid4

import pytest
from asapdiscovery.data.openeye import load_openeye_sdf
from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.data.schema_v2.ligand import Ligand, LigandIdentifiers
from asapdiscovery.data.testing.test_resources import fetch_test_file
from uuid import uuid4


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


def test_inchi(smiles):
    lig = Ligand.from_smiles(smiles)
    assert lig.inchi == "InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3"


def test_inchi_key(smiles):
    lig = Ligand.from_smiles(smiles)
    assert lig.inchikey == "IMNFDUFMRHMDMM-UHFFFAOYSA-N"


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name", None])
def test_ligand_dict_roundtrip(
    smiles,
    compound_name,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data,
):
    l1 = Ligand.from_smiles(
        smiles,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
        ),
        experimental_data=exp_data,
    )
    l2 = Ligand.from_dict(l1.dict())
    assert l1 == l2


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name", None])
def test_ligand_json_roundtrip(
    smiles,
    compound_name,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data,
):
    l1 = Ligand.from_smiles(
        smiles,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
        ),
        experimental_data=exp_data,
    )
    l2 = Ligand.from_json(l1.json())
    assert l1 == l2


def test_ligand_sdf_rountrip(moonshot_sdf, tmp_path):
    l1 = Ligand.from_sdf(moonshot_sdf)
    l1.to_sdf(tmp_path / "test.sdf")
    l2 = Ligand.from_sdf(tmp_path / "test.sdf")
    assert l1 == l2


@pytest.mark.parametrize(
    "exp_data", [ExperimentalCompoundData(compound_id="blah", smiles="CCCC"), None]
)  # FIXME this should be forced to match
@pytest.mark.parametrize("moonshot_compound_id", ["test_moonshot_compound_id", None])
@pytest.mark.parametrize("manifold_vc_id", ["ASAP-VC-1234", None])
@pytest.mark.parametrize("manifold_api_id", [uuid4(), None])
@pytest.mark.parametrize("compound_name", ["test_name", None])
def test_ligand_sdf_rountrip_data_only(
    moonshot_sdf,
    compound_name,
    manifold_api_id,
    manifold_vc_id,
    moonshot_compound_id,
    exp_data,
    tmp_path,
):
    l1 = Ligand.from_sdf(
        moonshot_sdf,
        compound_name=compound_name,
        ids=LigandIdentifiers(
            manifold_api_id=manifold_api_id,
            manifold_vc_id=manifold_vc_id,
            moonshot_compound_id=moonshot_compound_id,
        ),
        experimental_data=exp_data,
    )
    l1.to_sdf(tmp_path / "test.sdf")
    l2 = Ligand.from_sdf(tmp_path / "test.sdf")
    assert l1.data_equal(l2)


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


def test_get_set_sd_data(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf)
    l1.set_SD_data("test_key", "test_value")
    assert "> <test_key>" in l1.data
    assert "test_key" in l1.data
    assert l1.get_SD_data("test_key") == "test_value"


def test_print_sd_data(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf)
    l1.set_SD_data("test_key", "test_value")
    l1.print_SD_Data()


def test_get_set_sd_data_dict(moonshot_sdf):
    l1 = Ligand.from_sdf(moonshot_sdf)
    data = {"test_key": "test_value", "test_key2": "test_value2", "test_key3": "3"}
    l1.set_SD_data_dict(data)
    data_pulled = l1.get_SD_data_dict()
    assert data_pulled == data
