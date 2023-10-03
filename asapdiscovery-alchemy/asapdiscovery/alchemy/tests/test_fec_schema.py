import openfe
import pytest
from alchemiscale import Scope, ScopedKey
from asapdiscovery.alchemy.schema.atom_mapping import (
    KartografAtomMapper,
    LomapAtomMapper,
    PersesAtomMapper,
)
from asapdiscovery.alchemy.schema.fec import (
    AlchemiscaleResults,
    FreeEnergyCalculationFactory,
    SolventSettings,
    TransformationResult,
)
from asapdiscovery.alchemy.schema.network import NetworkPlanner
from openff.units import unit as OFFUnit


@pytest.mark.parametrize(
    "mapper, argument, value",
    [
        pytest.param(LomapAtomMapper, "max3d", 30, id="Lomap"),
        pytest.param(PersesAtomMapper, "coordinate_tolerance", 0.15, id="Perses"),
        pytest.param(
            KartografAtomMapper, "atom_ring_matches_ring", True, id="Kartograph"
        ),
    ],
)
def test_atom_mapper_settings(mapper, argument, value):
    """Make sure the settings are passed to the atom mapper object"""

    mapping_settings = mapper(**{argument: value})

    mapper_class = mapping_settings.get_mapper()
    assert getattr(mapper_class, argument) == getattr(mapping_settings, argument)


def test_lomap_atom_mapper_timeout():
    """Make sure the timeout setting is correctly passed to lomap as we have changed the naming."""

    mapper = LomapAtomMapper(timeout=50)
    engine = mapper.get_mapper()
    assert engine.time == mapper.timeout


@pytest.mark.parametrize(
    "mapper, programs",
    [
        pytest.param(LomapAtomMapper, ["openfe", "lomap", "rdkit"], id="Lomap"),
        pytest.param(
            PersesAtomMapper, ["openfe", "perses", "openeye.oechem"], id="Perses"
        ),
        pytest.param(
            KartografAtomMapper, ["openfe", "rdkit", "kartograf"], id="Kartograph"
        ),
    ],
)
def test_mapper_provenance(mapper, programs):
    """Make sure all used software are present in the provenance of the lomap atom mapper"""

    mapper_settings = mapper()
    provenance = mapper_settings.provenance()
    for program in programs:
        assert program in provenance


@pytest.mark.parametrize(
    "network_type",
    [
        pytest.param("radial", id="Radial"),
        pytest.param("maximal", id="Maximal"),
        pytest.param("minimal_spanning", id="Minimal Spanning"),
    ],
)
def test_network_planner_get_network(network_type):
    """Make sure we get the correct network planner based on the network_planning_method setting."""

    planner = NetworkPlanner(network_planning_method=network_type)

    planning_func = planner._get_network_plan()
    assert network_type in planning_func.__name__


@pytest.mark.parametrize(
    "scorer",
    [
        pytest.param("default_lomap", id="Lomap"),
        pytest.param("default_perses", id="Perses"),
    ],
)
def test_network_planner_get_scorer(scorer):
    """Make sure we get the correct atom mapping scoring method based on the scorer setting."""

    planner = NetworkPlanner(scorer=scorer)

    scoring_func = planner._get_scorer()
    assert scorer in scoring_func.__name__


@pytest.mark.parametrize(
    "network_type",
    [
        pytest.param("radial", id="Radial"),
        pytest.param("maximal", id="Maximal"),
        pytest.param("minimal_spanning", id="Minimal Spanning"),
    ],
)
def test_generate_network_lomap(network_type, tyk2_ligands):
    """Test generating ligand FEC networks with the configured settings using lomap."""

    if network_type == "radial":
        central = tyk2_ligands[0]
        ligands = tyk2_ligands[1:]
    else:
        central = None
        ligands = tyk2_ligands
    # configure the mapper
    planner = NetworkPlanner(
        atom_mapping_engine=LomapAtomMapper(),
        scorer="default_lomap",
        network_planning_method=network_type,
    )

    planned_network = planner.generate_network(ligands=ligands, central_ligand=central)

    fe_network = planned_network.to_ligand_network()
    # make sure we have all the ligands we expect
    assert len(fe_network.nodes) == 10
    if network_type == "radial":
        # radial should have all ligands connected to the central node
        assert len(fe_network.edges) == 9

    # make sure we can convert back to openfe ligands
    openfe_ligands = planned_network.to_openfe_ligands()
    assert len(openfe_ligands) == 10
    assert isinstance(openfe_ligands[0], openfe.SmallMoleculeComponent)


def test_plan_radial_error(tyk2_ligands):
    """Make sure an error is raised if we try and plan a radial network with no central ligand"""
    planner = NetworkPlanner(network_planning_method="radial")
    with pytest.raises(RuntimeError):
        _ = planner.generate_network(ligands=tyk2_ligands)


def test_solvent_settings():
    """Make sure solvent settings are correctly passed to the gufe solvent component."""

    settings = SolventSettings()
    settings.ion_concentration = 0.25 * OFFUnit.molar

    component = settings.to_solvent_component()
    # make sure they match with units
    assert component._ion_concentration == settings.ion_concentration
    # check the magnitude
    assert component._ion_concentration.m == 0.25


def test_planner_file_round_trip(tmpdir):
    """Make sure we can serialise a network planner to and from file."""

    with tmpdir.as_cwd():
        # configure with non default settings
        filename = "network_planner.json"
        planner = NetworkPlanner(scorer="default_perses")
        planner.to_file(filename=filename)
        planner_2 = NetworkPlanner.from_file(filename=filename)
        assert planner.scorer == planner_2.scorer


def test_fec_to_openfe_protocol():
    """Make sure we can correctly reconstruct the openfe protocol needed to run the calculation from the factory settings"""

    # change some default settings to make sure they are passed on
    factory = FreeEnergyCalculationFactory()
    factory.simulation_settings.equilibration_length = 0.5 * OFFUnit.nanoseconds
    protocol = factory.to_openfe_protocol()
    assert isinstance(
        protocol, openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol
    )
    assert (
        protocol.settings.simulation_settings.equilibration_length
        == factory.simulation_settings.equilibration_length
    )


def test_fec_dataset_duplicate_ligands(tyk2_ligands, tyk2_protein):
    # duplicate a ligand
    ligands = tyk2_ligands[-1:] + tyk2_ligands

    factory = FreeEnergyCalculationFactory()
    with pytest.raises(ValueError, match="1 duplicate ligands"):
        _ = factory.create_fec_dataset(
            dataset_name="TYK2-test-dataset-duplicated",
            receptor=tyk2_protein,
            ligands=ligands,
        )

    ligands = ligands + tyk2_ligands[:1]

    factory = FreeEnergyCalculationFactory()
    with pytest.raises(ValueError, match="2 duplicate ligands"):
        _ = factory.create_fec_dataset(
            dataset_name="TYK2-test-dataset-duplicated",
            receptor=tyk2_protein,
            ligands=ligands,
        )


def test_fec_full_workflow(tyk2_ligands, tyk2_protein):
    """Make sure we can run the full FEC workflow"""
    factory = FreeEnergyCalculationFactory()
    # change the default settings to make sure they propagated
    # change the lomap timeout
    factory.network_planner.atom_mapping_engine.timeout = 30
    factory.simulation_settings.equilibration_length = 0.5 * OFFUnit.nanoseconds
    # plan a network
    planned_network = factory.create_fec_dataset(
        dataset_name="TYK2-test-dataset", receptor=tyk2_protein, ligands=tyk2_ligands
    )
    # make sure the settings were used correctly
    assert planned_network.network.atom_mapping_engine.timeout == 30
    assert "openfe" in planned_network.network.provenance
    # make sure we can rebuild the receptor
    _ = planned_network.to_openfe_receptor()
    # make sure we can build an openfe alchemical network
    alchemical_network = planned_network.to_alchemical_network()
    # make sure the equilibration time was updated
    for edge in alchemical_network.edges:
        assert (
            edge.protocol.settings.simulation_settings.equilibration_length
            == 0.5 * OFFUnit.nanoseconds
        )


def test_results_to_cinnabar_missing_phase(tyk2_fec_network):
    """Make sure an error is raised if we try and convert to a cinnabar results with missing simulated phases."""

    alchem_network = tyk2_fec_network.to_alchemical_network()
    results = []
    # mock some results for only the complex phase
    for edge in alchem_network.edges:
        if "complex" in edge.name:
            results.append(
                TransformationResult(
                    ligand_a=edge.stateA.components["ligand"].name,
                    ligand_b=edge.stateB.components["ligand"].name,
                    phase="complex",
                    estimate=1 * OFFUnit.kilocalorie / OFFUnit.mole,
                    uncertainty=0 * OFFUnit.kilocalorie / OFFUnit.mole,
                )
            )
    # mock a full result object
    scope = Scope(org="asap", campaign="testing", project="tyk2")
    result_network = AlchemiscaleResults(
        network_key=ScopedKey(gufe_key=alchem_network.key, **scope.dict()),
        results=results,
    )
    # make sure a specific error related to a missing solvent phase is raised.
    with pytest.raises(
        RuntimeError,
        match="is missing simulated legs in the following phases {'solvent'}",
    ):
        result_network.to_fe_map()


def test_results_to_cinnabar_too_many_legs(tyk2_fec_network):
    """Make sure an error is raised if we have too many results for a transformation when trying to convet to cinnabar."""

    alchem_network = tyk2_fec_network.to_alchemical_network()
    results = []
    # mock some results for only the complex phase
    for edge in alchem_network.edges:
        if "complex" in edge.name:
            phase = "complex"
        else:
            phase = "solvent"

        transform_result = TransformationResult(
            ligand_a=edge.stateA.components["ligand"].name,
            ligand_b=edge.stateB.components["ligand"].name,
            phase=phase,
            estimate=1 * OFFUnit.kilocalorie / OFFUnit.mole,
            uncertainty=0 * OFFUnit.kilocalorie / OFFUnit.mole,
        )
        # if solvent phase add twice
        if phase == "complex":
            results.append(transform_result)
        else:
            results.extend([transform_result, transform_result])

    # mock a full result object
    scope = Scope(org="asap", campaign="testing", project="tyk2")
    result_network = AlchemiscaleResults(
        network_key=ScopedKey(gufe_key=alchem_network.key, **scope.dict()),
        results=results,
    )
    # make sure a specific error related to a missing solvent phase is raised.
    with pytest.raises(
        RuntimeError, match="has too many simulated legs, found the following phases"
    ):
        result_network.to_fe_map()
