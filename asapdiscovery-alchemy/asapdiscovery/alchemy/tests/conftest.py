import openfe
import pytest
from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
from asapdiscovery.alchemy.utils import AlchemiscaleHelper
from asapdiscovery.data.testing.test_resources import fetch_test_file
from rdkit import Chem


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


@pytest.fixture()
def mock_alchemiscale_client(monkeypatch):
    """Mock alchemiscale client for testing purposes"""
    # mock some env variables
    monkeypatch.setenv(name="ALCHEMISCALE_ID", value="asap")
    monkeypatch.setenv(name="ALCHEMISCALE_KEY", value="key")

    # use a fake api url for testing
    client = AlchemiscaleHelper(api_url="")

    return client