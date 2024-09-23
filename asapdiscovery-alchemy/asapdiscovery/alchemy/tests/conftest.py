import datetime
import tempfile

import openfe
import pytest
from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
from asapdiscovery.alchemy.schema.prep_workflow import AlchemyPrepWorkflow
from asapdiscovery.alchemy.utils import AlchemiscaleHelper
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.schema.complex import PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand, write_ligands_to_multi_sdf
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.schema.pose_generation import OpenEyeConstrainedPoseGenerator
from gufe.protocols import Context, ProtocolUnit, ProtocolUnitFailure


@pytest.fixture(scope="session")
def tyk2_ligands():
    """Create a set of openfe tyk2 ligands"""
    input_ligands = fetch_test_file("tyk2_ligands.sdf")
    ligands = MolFileFactory(filename=input_ligands.as_posix()).load()
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


@pytest.fixture(scope="session")
def tyk2_result_network():
    """Return an FEC network with some results."""
    fec_network = fetch_test_file("tyk2_result_network.json")
    return FreeEnergyCalculationNetwork.from_file(fec_network)


@pytest.fixture(scope="session")
def tyk2_result_network_disconnected():
    """Return an FEC network with some results."""
    fec_network = fetch_test_file("tyk2_result_network_disconnected.json")
    return FreeEnergyCalculationNetwork.from_file(fec_network)


@pytest.fixture(scope="session")
def tyk2_reference_data():
    """Return a CSV in the CDD style of IC50 values for the tyk2 series."""
    return fetch_test_file("tyk2_reference_data.csv")


@pytest.fixture(scope="session")
def tyk2_small_custom_network():
    """The path to a csv file which can be used to plan a tyk2 network."""
    return fetch_test_file("tyk2_small_custom_network.csv")


@pytest.fixture(scope="session")
def tyk2_small_custom_network_faulty_missing_comma():
    return fetch_test_file("tyk2_small_custom_network_faulty_missing_comma.csv")


@pytest.fixture(scope="session")
def tyk2_small_custom_network_faulty_with_spaces():
    return fetch_test_file("tyk2_small_custom_network_faulty_with_spaces.csv")


@pytest.fixture(scope="function")
def alchemiscale_helper():
    # use a fake api url for testing
    client = AlchemiscaleHelper(api_url="", key="key", identifier="asap")

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


@pytest.fixture(scope="session")
def mac1_complex():
    return PreppedComplex.parse_file(
        fetch_test_file("constrained_conformer/complex.json")
    )


@pytest.fixture()
def openeye_charged_prep_workflow() -> AlchemyPrepWorkflow:
    """Build an openeye pose generator for testing as its faster than rdkit."""
    return AlchemyPrepWorkflow(
        charge_expander=None, pose_generator=OpenEyeConstrainedPoseGenerator()
    )


@pytest.fixture()
def openeye_prep_workflow() -> AlchemyPrepWorkflow:
    return AlchemyPrepWorkflow(
        charge_expander=None,
        pose_generator=OpenEyeConstrainedPoseGenerator(),
        charge_method=None,
    )


@pytest.fixture()
def test_ligands():
    TEST_LIGANDS = [
        Ligand.from_smiles(smi, compound_name="foo")
        for smi in [
            "O=C(NC1=CC(Cl)=CC(C(=O)NC2=CC=C(CC3CCNCC3)C=C2)=C1)OCC1=CC=CC=C1",
            "CCNC(=O)NC1=CC(Cl)=CC(C(=O)NC2=CC(C)=CC(CN)=C2)=C1",
            "NC1=CC=C(NC(=O)C2=CC(Cl)=CC3=C2C=NN3)C=N1",
            "NCC1=CC=CC(NC(=O)C2=CC(Cl)=CC(CN)=C2)=C1",
            "O=C(C1=CC=CC2=CC=CC=C12)NC3=CC=C4CNCC4=C3",
            "CCNC(=O)NC1=CC(Cl)=CC(C(=O)NC2=CC(C)=CC(CN)=C2)=C1",
            "O=C(C1=CC=CC2=C(F)C=CC=C12)NC3=CC=C4CNCC4=C3",
            "O=C(C1=CC=CC2=C(Cl)C=CC=C12)NC3=CC=C4CNCC4=C3",
            "O=C(C1=CC=CC2=C(Br)C=CC=C12)NC3=CC=C4CNCC4=C3",
        ]
    ]
    return TEST_LIGANDS


@pytest.fixture()
def test_ligands_sdfile(test_ligands, tmp_path):
    # write the ligands to a temporary SDF file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sdf", delete=False, dir=tmp_path
    ) as f:
        write_ligands_to_multi_sdf(f.name, test_ligands, overwrite=True)
    return f.name


@pytest.fixture()
def tyk2_result_network_ddg0s():
    return fetch_test_file("tyk2_result_network_ddg0s.json")


@pytest.fixture()
def p38_graphml():
    return fetch_test_file("p38.graphml")


@pytest.fixture()
def p38_protein():
    return fetch_test_file("p38.pdb")
