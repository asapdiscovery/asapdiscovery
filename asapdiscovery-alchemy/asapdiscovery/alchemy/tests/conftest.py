import datetime

import openfe
import pytest
from gufe.protocols import Context, ProtocolUnit, ProtocolUnitFailure
from rdkit import Chem

from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
from asapdiscovery.alchemy.utils import AlchemiscaleHelper
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def tyk2_ligands():
    """Create a set of openfe tyk2 ligands"""
    input_ligands = fetch_test_file("tyk2_ligands.sdf")
    supp = Chem.SDMolSupplier(input_ligands.as_posix(), removeHs=False)
    # convert to openfe objects
    ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]
    return ligands


@pytest.fixture(scope="session")
def tyk2_protein():
    """Create an openfe protein component"""
    receptor_file = fetch_test_file("tyk2_protein.pdb")
    return openfe.ProteinComponent.from_pdb_file(receptor_file.as_posix())


@pytest.fixture(scope="session")
def tyk2_fec_network():
    """Create a FEC planned network from file"""
    fec_network = fetch_test_file("tyk2_small_network.json")
    return FreeEnergyCalculationNetwork.from_file(fec_network)


@pytest.fixture(scope="function")
def alchemiscale_helper(monkeypatch):
    monkeypatch.setenv(name="ALCHEMISCALE_ID", value="asap")
    monkeypatch.setenv(name="ALCHEMISCALE_KEY", value="key")

    # use a fake api url for testing
    client = AlchemiscaleHelper(api_url="")

    # make sure the env variables were picked up
    assert client._client.identifier == "asap"
    assert client._client.key == "key"

    return client


# Test gufe "fixtures"
class DummyUnit(ProtocolUnit):
    @staticmethod
    def _execute(ctx: Context, an_input=2, **inputs):
        if an_input != 2:
            raise ValueError("`an_input` should always be 2(!!!)")

        return {"foo": "bar"}


@pytest.fixture
def dummy_protocol_units() -> list[ProtocolUnit]:
    """Create list of 3 Dummy protocol units"""
    units = [DummyUnit(name=f"dummy{i}") for i in range(3)]
    return units


@pytest.fixture()
def protocol_unit_failures(dummy_protocol_units) -> list[list[ProtocolUnitFailure]]:
    """generate 2 unit failures for every protocol unit"""
    t1 = datetime.datetime.now()
    t2 = datetime.datetime.now()

    return [
        [
            ProtocolUnitFailure(
                source_key=u.key,
                inputs=u.inputs,
                outputs=dict(),
                exception=("ValueError", ("Didn't feel like it",)),
                traceback="foo",
                start_time=t1,
                end_time=t2,
            )
            for _ in range(2)
        ]
        for u in dummy_protocol_units
    ]
