import itertools
from uuid import uuid4

import pandas
import pytest
from alchemiscale import Scope, ScopedKey
from asapdiscovery.alchemy.cli.utils import get_cdd_molecules, upload_to_postera
from asapdiscovery.alchemy.schema.fec import (
    AlchemiscaleResults,
    FreeEnergyCalculationNetwork,
)
from asapdiscovery.alchemy.utils import extract_custom_ligand_network
from gufe.protocols import ProtocolDAGResult, ProtocolUnitResult
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
    status = client.network_status(network_key=result_network.results.network_key)
    assert status == {"complete": 1}


def test_action_tasks(monkeypatch, tyk2_fec_network, alchemiscale_helper):
    """Make sure the helper can action tasks on alchemiscale."""

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
            ScopedKey(gufe_key=f"task-{uuid4().hex}", **transformation.scope.dict())
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
    task_replicas = 3  # the total number of times each task should be run
    tasks = client.action_network(planned_network=result_network, repeats=task_replicas)

    assert len(tasks) == (task_replicas * len(alchem_network.edges))


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

    def get_network_results(*args, **kwargs):
        """Mock pull down the results for all transforms in a network"""
        results = {}
        for key, edge in keys_to_edges.items():
            if "complex" in edge.name:
                estimate = 3
            else:
                estimate = 1

            task_result = ProtocolUnitResult(
                name=edge.name,
                source_key=key,
                inputs={"stateA": edge.stateA, "stateB": edge.stateB},
                outputs={
                    "unit_estimate": estimate * OFFUnit.kilocalorie / OFFUnit.mole,
                    "unit_estimate_error": 0.1 * OFFUnit.kilocalorie / OFFUnit.mole,
                },
            )
            results[key] = RelativeHybridTopologyProtocolResult(
                **{edge.name: [task_result]}
            )

        return results

    def get_transformation(*args, **kwargs):
        """Mock pulling down the transformation"""
        transform_key = kwargs["transformation"]
        return keys_to_edges[transform_key]


    # mock the collection api
    monkeypatch.setattr(client._client, "get_network_results", get_network_results)
    monkeypatch.setattr(client._client, "get_transformation", get_transformation)

    network_with_results = client.collect_results(planned_network=result_network)
    # convert to cinnabar fep map
    result_map = network_with_results.results.to_fe_map()
    assert (
        result_map.n_ligands == len(alchem_network.nodes) / 2
    )  # divide by 2 as we have a node for the solvent and complex phase


def test_restart_tasks(monkeypatch, tyk2_fec_network, alchemiscale_helper):
    client = alchemiscale_helper

    scope = Scope(org="asap", campaign="testing", project="tyk2")
    alchemical_network = tyk2_fec_network.to_alchemical_network()

    network_key = ScopedKey(gufe_key=alchemical_network.key, **scope.dict())
    task_keys = [
        ScopedKey(gufe_key=f"task-{uuid4().hex}", **network_key.scope.dict()) for _ in range(7)
    ]

    def get_network_tasks(key: ScopedKey, status: str):
        assert key == network_key
        # pretend 7 tasks have status == 'error'
        return task_keys

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

    # restart only a subset of the errored tasks
    errored_tasks = client._client.get_network_tasks(network_key, status="error")
    restarted_tasks = client.restart_tasks(
        planned_network=result_network, tasks=errored_tasks[:2]
    )

    assert len(restarted_tasks) == 2
    assert [isinstance(i, ScopedKey) for i in restarted_tasks]
    assert set(restarted_tasks) == set(errored_tasks[:2])


def test_get_failures(
    monkeypatch,
    tyk2_fec_network,
    alchemiscale_helper,
    dummy_protocol_units,
    protocol_unit_failures,
):
    """Make sure we can get exceptions and tracebacks from failures in a network"""

    # use a fake api url for testing
    client = alchemiscale_helper

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
    def get_network_tasks(key, status=None) -> list[ScopedKey]:
        """Mock getting back a single Task for each Transformation in an AlchemicalNetwork"""
        assert key == network_key
        tasks = []
        for edge in alchem_network.edges:
            tf_key = edge.key
            task_key = tf_key.replace("Transformation", "Task")
            # 1 task per edge -- 18 tasks in total
            tasks.append(ScopedKey(gufe_key=task_key, **scope.dict()))
        return tasks

    def get_task_failures(key) -> list[ProtocolDAGResult]:
        """Mock pulling the Task failures from alchemiscale"""
        dagresult = ProtocolDAGResult(
            protocol_units=dummy_protocol_units,  # 3 dummy units per dag result
            protocol_unit_results=list(itertools.chain(*protocol_unit_failures)),
            transformation_key=None,
        )
        task_failures = [dagresult]
        return task_failures

    # mock the status function
    monkeypatch.setattr(client._client, "get_network_tasks", get_network_tasks)
    monkeypatch.setattr(client._client, "get_task_failures", get_task_failures)

    # Collect errors and tracebacks
    errors = client.collect_errors(network_key=result_network.results.network_key)
    n_errors = len(errors)
    n_expected_errors = (
        108  # 18*3*2 (18 tasks, 1 dag/task, 3 units/dag, 2 failures/unit)
    )
    assert (
        n_errors == n_expected_errors
    ), f"Expected {n_expected_errors} errors, received {n_errors} errors."
    # Check tracebacks
    assert all(
        "foo" == data.traceback for data in errors
    ), "'foo' string expected in `traceback` attribute."


def test_get_actioned_weights(alchemiscale_helper, monkeypatch, tyk2_fec_network):
    """Mock getting the network weights for submitted networks"""

    client = alchemiscale_helper

    scope = Scope(org="asap", campaign="testing", project="tyk2")
    network_key = ScopedKey(
        gufe_key=tyk2_fec_network.to_alchemical_network().key, **scope.dict()
    )

    def query_networks(*args, **kwargs):
        # return many networks to trigger the early stopping
        return [network_key for _ in range(10)]

    def network_status(network, *args, **kwargs):
        assert network == network_key
        return {"complete": 1, "error": 1}

    def network_weight(network):
        assert network == network_key
        return 0.5

    monkeypatch.setattr(client._client, "query_networks", query_networks)
    monkeypatch.setattr(client._client, "get_network_status", network_status)
    monkeypatch.setattr(client._client, "get_network_weight", network_weight)

    active_network_weights = client.get_actioned_weights()
    # we should have 4 weights from early stopping
    assert active_network_weights == [0.5 for _ in range(4)]


def test_upload_to_postera(monkeypatch, tyk2_result_network):
    """A mocked test to make sure the dataframe is formatted correctly ready for upload to postera."""
    from asapdiscovery.alchemy.predict import get_data_from_femap
    from asapdiscovery.data.services.postera.postera_uploader import PosteraUploader

    fe_map = tyk2_result_network.results.to_fe_map()
    fe_map.generate_absolute_values()

    absolute_df, _ = get_data_from_femap(
        fe_map=fe_map,
        ligands=tyk2_result_network.network.ligands,
        assay_units=None,
        reference_dataset=None,
        cdd_protocol=None,
    )

    def push(*args, **kwargs):
        df = kwargs["df"]
        # check we have the expected column names
        assert (
            "biochemical-activity_SARS-CoV-2-Mpro_computed-FEC-pIC50_msk" in df.columns
        )
        assert (
            "biochemical-activity_SARS-CoV-2-Mpro_computed-FEC-uncertainty-pIC50_msk"
            in df.columns
        )
        assert "Ligand_ID" in df.columns
        assert "SMILES" in df.columns
        return df, 1, False

    monkeypatch.setenv("POSTERA_API_KEY", "my-key")
    monkeypatch.setattr(PosteraUploader, "push", push)

    upload_to_postera(
        molecule_set_name="my_mol_set",
        target="SARS-CoV-2-Mpro",
        absolute_dg_predictions=absolute_df,
    )


@pytest.mark.parametrize(
    "defined_only, n_ligands, remove_covalent",
    [
        pytest.param(True, 1, True, id="Defined only no warhead."),
        pytest.param(False, 2, True, id="All ligands no warhead."),
        pytest.param(True, 2, False, id="Defined only with warhead"),
        pytest.param(False, 3, False, id="All ligands with warhead."),
    ],
)
def test_get_cdd_molecules_util(monkeypatch, defined_only, n_ligands, remove_covalent):
    """Test downloading molecules from a cdd mocked protocol and removing molecules with undefined stereo."""

    import asapdiscovery.alchemy.predict

    def get_cdd_data(protocol_name: str):
        data = [
            {"Smiles": "CCO", "Molecule Name": "ethanol", "CXSmiles": "CCO"},
            {
                "Smiles": "NC(C)C(=O)O",
                "Molecule Name": "alanine",
                "CXSmiles": "NC(C)C(=O)O",
            },
            {
                "Smiles": r"CCOC1=C(NC(=O)\C=C\CN(C)C)C=C2C(NC3=CC=C(OCC4=CC=CC=N4)C(Cl)=C3)=C(C=NC2=C1)C#N",
                "Molecule Name": "Neratinib",
                "CXSmiles": r"CCOC1=C(NC(=O)\C=C\CN(C)C)C=C2C(NC3=CC=C(OCC4=CC=CC=N4)C(Cl)=C3)=C(C=NC2=C1)C#N",
            },
        ]
        return pandas.DataFrame(data)

    monkeypatch.setattr(
        asapdiscovery.alchemy.predict, "download_cdd_data", get_cdd_data
    )

    molecules = get_cdd_molecules(
        protocol_name="my-protocol",
        defined_stereo_only=defined_only,
        remove_covalent=remove_covalent,
    )
    assert len(molecules) == n_ligands
    # check they are marked as experimental
    for ligand in molecules:
        assert ligand.tags["experimental"] == "True"
        assert ligand.tags["cdd_protocol"] == "my-protocol"


def test_cdd_download_remove_radicals(monkeypatch):
    """Make sure radical molecules are removed from downloaded protocols."""

    import asapdiscovery.alchemy.predict

    def get_cdd_data(protocol_name: str):
        data = [
            {"Smiles": "CCO", "Molecule Name": "ethanol", "CXSmiles": "CCO"},
            {
                "Smiles": "C[CH2]",
                "Molecule Name": "radical",
                "CXSmiles": "C[CH2]",
            },
        ]
        return pandas.DataFrame(data)

    monkeypatch.setattr(
        asapdiscovery.alchemy.predict, "download_cdd_data", get_cdd_data
    )

    molecules = get_cdd_molecules(
        protocol_name="my-protocol",
        defined_stereo_only=True,
        remove_covalent=True,
    )
    assert len(molecules) == 1
    assert molecules[0].compound_name == "ethanol"


def test_extracting_invalid_transforms(tyk2_small_custom_network_faulty_missing_comma):
    """Make sure an error is raised if the file is not formated correctly."""
    with pytest.raises(
        ValueError, match="Custom network file contains an empty entry at index 2"
    ):
        extract_custom_ligand_network(tyk2_small_custom_network_faulty_missing_comma)


def test_extracting_transforms_wtih_spaces(
    tyk2_small_custom_network_faulty_with_spaces,
):
    """Make sure we can extract names even with poor formatting."""

    edges = extract_custom_ligand_network(tyk2_small_custom_network_faulty_with_spaces)
    for edge in edges:
        for ligand_name in edge:
            assert " " not in ligand_name
