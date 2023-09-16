from uuid import uuid4

from alchemiscale import Scope, ScopedKey
from asapdiscovery.alchemy.schema.fec import (
    AlchemiscaleResults,
    FreeEnergyCalculationNetwork,
)
from gufe.protocols import ProtocolUnitResult
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocolResult
from openff.units import unit as OFFUnit


def test_create_network(monkeypatch, tyk2_fec_network, alchemiscale_helper):
    """Make sure our alchemiscale helper can create a network and update the results"""

    client = alchemiscale_helper

    # mock the client function
    def create_network(network, scope):
        return ScopedKey(gufe_key=network.key, **scope.dict())

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


def test_network_status(monkeypatch, tyk2_fec_network, alchemiscale_helper):
    """Make sure we can correctly get the status of a network"""

    client = alchemiscale_helper

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


def test_action_tasks(monkeypatch, tyk2_fec_network, alchemiscale_helper):
    """Make sure the helper can action tasks on alchemiscale"""

    client = alchemiscale_helper

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
        return [
            ScopedKey(gufe_key=uuid4().hex, **transformation.scope.dict())
            for _ in range(count)
        ]

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


def test_collect_results(monkeypatch, tyk2_fec_network, alchemiscale_helper):
    """Make sure the help function can correctly collect results"""

    client = alchemiscale_helper

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


def test_restart_tasks(monkeypatch, tyk2_fec_network, alchemiscale_helper):
    client = alchemiscale_helper

    scope = Scope(org="asap", campaign="testing", project="tyk2")
    alchemical_network = tyk2_fec_network.to_alchemical_network()

    network_key = ScopedKey(gufe_key=alchemical_network.key, **scope.dict())

    def get_network_tasks(key: ScopedKey, status: str):
        assert key == network_key
        # pretend 7 tasks have status == 'error'
        return [ScopedKey(gufe_key=uuid4().hex, **key.scope.dict()) for _ in range(7)]

    def set_tasks_status(tasks: list[ScopedKey], status: str):
        return tasks

    # mock the AlchemiscaleClient calls
    monkeypatch.setattr(client._client, "get_network_tasks", get_network_tasks)
    monkeypatch.setattr(client._client, "set_tasks_status", set_tasks_status)

    # set the key and get the status
    result_network = FreeEnergyCalculationNetwork(
        **tyk2_fec_network.dict(exclude={"results"}),
        results=AlchemiscaleResults(network_key=network_key),
    )

    restarted_tasks = client.restart_tasks(planned_network=result_network)
    assert len(restarted_tasks) == 7
    assert [isinstance(i, ScopedKey) for i in restarted_tasks]
