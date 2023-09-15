import datetime
import itertools

import pytest
from alchemiscale import Scope, ScopedKey
from asapdiscovery.simulation.schema.fec import (
    AlchemiscaleResults,
    FreeEnergyCalculationNetwork,
)
from gufe.protocols import (
    Context,
    ProtocolDAGResult,
    ProtocolUnit,
    ProtocolUnitFailure,
    ProtocolUnitResult,
)
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocolResult
from openff.units import unit as OFFUnit


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
    """generate 2 unit failures for every task"""
    t1 = datetime.datetime.now()
    t2 = datetime.datetime.now()

    return [
        [
            ProtocolUnitFailure(
                source_key=u.key,
                inputs=u.inputs,
                outputs=dict(),
                exception=("ValueError", "Didn't feel like it"),
                traceback="foo",
                start_time=t1,
                end_time=t2,
            )
            for _ in range(2)
        ]
        for u in dummy_protocol_units
    ]


def test_create_network(monkeypatch, tyk2_fec_network, mock_alchemiscale_client):
    """Make sure our alchemiscale helper can create a network and update the results"""

    # mock the client function
    def create_network(network, scope):
        return ScopedKey(gufe_key=network.key, **scope.dict())

    # use a fake api url for testing
    client = mock_alchemiscale_client

    # make sure the env variables were picked up
    assert client._client.identifier == "asap"
    assert client._client.key == "key"

    # mock the network creation
    monkeypatch.setattr(client._client, "create_network", create_network)
    scope = Scope(org="asap", campaign="testing", project="tyk2")
    assert tyk2_fec_network.results is None
    result = client.create_network(planned_network=tyk2_fec_network, scope=scope)
    # make sure the results have been updated
    assert isinstance(result.results, AlchemiscaleResults)
    # make sure the network key has been saved
    assert result.results.network_key == ScopedKey(
        gufe_key=tyk2_fec_network.to_alchemical_network().key, **scope.dict()
    )


def test_network_status(monkeypatch, tyk2_fec_network, mock_alchemiscale_client):
    """Make sure we can correctly get the status of a network"""

    client = mock_alchemiscale_client

    scope = Scope(org="asap", campaign="testing", project="tyk2")
    network_key = ScopedKey(
        gufe_key=tyk2_fec_network.to_alchemical_network().key, **scope.dict()
    )

    def network_status(key: str):
        assert key == network_key
        return {"complete": 1}

    # mock the status function
    monkeypatch.setattr(client._client, "get_network_status", network_status)

    # set the key and get the status
    result_network = FreeEnergyCalculationNetwork(
        **tyk2_fec_network.dict(exclude={"results"}),
        results=AlchemiscaleResults(network_key=network_key),
    )
    status = client.network_status(planned_network=result_network)
    assert status == {"complete": 1}


def test_action_tasks(monkeypatch, tyk2_fec_network, mock_alchemiscale_client):
    """Make sure the helper can action tasks on alchemiscale"""

    # use a fake api url for testing
    client = mock_alchemiscale_client

    # mock a key onto the network assuming it has already been created
    scope = Scope(org="asap", campaign="testing", project="tyk2")
    network_key = ScopedKey(
        gufe_key=tyk2_fec_network.to_alchemical_network().key, **scope.dict()
    )
    alchem_network = tyk2_fec_network.to_alchemical_network()
    result_network = FreeEnergyCalculationNetwork(
        **tyk2_fec_network.dict(exclude={"results"}),
        results=AlchemiscaleResults(network_key=network_key),
    )

    # mock the client functions
    def get_network_transformations(key) -> list[ScopedKey]:
        """Mock pulling the transforms from alchemiscale"""
        assert key == network_key
        transforms = []
        for edge in alchem_network.edges:
            transforms.append(ScopedKey(gufe_key=edge.key, **scope.dict()))
        return transforms

    def create_tasks(transformation, count):
        "Mock creating tasks for a transform"
        return [transformation for _ in range(count)]

    def action_tasks(tasks, network):
        "mock actioning tasks"
        # make sure we get the correct key for the submission
        assert network == network_key
        return tasks

    monkeypatch.setattr(
        client._client, "get_network_transformations", get_network_transformations
    )
    monkeypatch.setattr(client._client, "create_tasks", create_tasks)
    monkeypatch.setattr(client._client, "action_tasks", action_tasks)

    tasks = client.action_network(planned_network=result_network)

    assert len(tasks) == (result_network.n_repeats + 1) * len(alchem_network.edges)


def test_collect_results(monkeypatch, tyk2_fec_network, mock_alchemiscale_client):
    """Make sure the help function can correctly collect results"""

    # use a fake api url for testing
    client = mock_alchemiscale_client

    # mock a key onto the network assuming it has already been created
    scope = Scope(org="asap", campaign="testing", project="tyk2")
    network_key = ScopedKey(
        gufe_key=tyk2_fec_network.to_alchemical_network().key, **scope.dict()
    )
    alchem_network = tyk2_fec_network.to_alchemical_network()
    result_network = FreeEnergyCalculationNetwork(
        **tyk2_fec_network.dict(exclude={"results"}),
        results=AlchemiscaleResults(network_key=network_key),
    )
    # track the keys to the transforms
    keys_to_edges = {
        ScopedKey(gufe_key=edge.key, **scope.dict()): edge
        for edge in alchem_network.edges
    }

    def get_network_transformations(key) -> list[ScopedKey]:
        """Mock pulling the transforms from alchemiscale"""
        assert key == network_key
        transforms = []
        for edge in alchem_network.edges:
            transforms.append(ScopedKey(gufe_key=edge.key, **scope.dict()))
        return transforms

    def get_transformation_results(task_key):
        """Mock pulling a result for a transform"""
        # create a specific result corresponding to the edge
        edge = keys_to_edges[task_key]
        if "complex" in edge.name:
            estimate = 3
        else:
            estimate = 1
        task_result = ProtocolUnitResult(
            name=edge.name,
            source_key=task_key,
            inputs={"stateA": edge.stateA, "stateB": edge.stateB},
            outputs={"unit_estimate": estimate * OFFUnit.kilocalorie / OFFUnit.mole},
        )
        return RelativeHybridTopologyProtocolResult(**{edge.name: [task_result]})

    # mock the collection api
    monkeypatch.setattr(
        client._client, "get_network_transformations", get_network_transformations
    )
    monkeypatch.setattr(
        client._client, "get_transformation_results", get_transformation_results
    )

    network_with_results = client.collect_results(planned_network=result_network)
    # convert to cinnabar fep map
    result_map = network_with_results.results.to_fe_map()
    assert (
        len(result_map.graph.nodes) == len(alchem_network.nodes) / 2
    )  # divide by 2 as we have a node for the solvent and complex phase


def test_get_failures(
    monkeypatch,
    tyk2_fec_network,
    mock_alchemiscale_client,
    dummy_protocol_units,
    protocol_unit_failures,
):
    """Make sure we can get exceptions and tracebacks from failures in a network"""

    # use a fake api url for testing
    client = mock_alchemiscale_client

    scope = Scope(org="asap", campaign="testing", project="tyk2")
    network_key = ScopedKey(
        gufe_key=tyk2_fec_network.to_alchemical_network().key, **scope.dict()
    )

    alchem_network = tyk2_fec_network.to_alchemical_network()
    # set the key and get the status
    result_network = FreeEnergyCalculationNetwork(
        **tyk2_fec_network.dict(exclude={"results"}),
        results=AlchemiscaleResults(network_key=network_key),
    )

    # mock the client functions
    def get_network_tasks(key, status=None) -> list[ScopedKey]:
        """Mock pulling the transforms from alchemiscale"""
        assert key == network_key
        tasks = []
        for edge in alchem_network.edges:
            tf_key = edge.key
            task_key = tf_key.replace("Transformation", "Task")
            tasks.append(ScopedKey(gufe_key=task_key, **scope.dict()))
        return tasks

    def get_task_failures(key) -> list[ProtocolDAGResult]:
        """Mock pulling the task failures from alchemiscale"""
        dagresult = ProtocolDAGResult(
            protocol_units=dummy_protocol_units,
            protocol_unit_results=list(itertools.chain(*protocol_unit_failures)),
            transformation_key=None,
        )
        task_failures = [dagresult]
        return task_failures

    # mock the status function
    monkeypatch.setattr(client._client, "get_network_tasks", get_network_tasks)
    monkeypatch.setattr(client._client, "get_task_failures", get_task_failures)

    # Collect errors without traceback
    errors = client.collect_errors(planned_network=result_network, with_traceback=False)
    n_errors = len(errors)
    assert n_errors == 18, f"Expected 18 errors, received {n_errors} errors."

    # With complete traceback
    errors = client.collect_errors(planned_network=result_network, with_traceback=True)
    assert n_errors == 18, f"Expected 18 errors, received {n_errors} errors."
    assert all(
        "traceback" in data.keys() for data in errors.values()
    ), "`traceback` key expected and not found in results dictionary."
