import openfe
import pytest
from asapdiscovery.simulation.schema.atom_mapping import (
    KartografAtomMapper,
    LomapAtomMapper,
    PersesAtomMapper,
)
from asapdiscovery.simulation.schema.fec import (
    FreeEnergyCalculationFactory,
    SolventSettings,
)
from asapdiscovery.simulation.schema.network import NetworkPlanner
from openff.units import unit as OFFUnit


@pytest.mark.parametrize(
    "mapper, argument, value",
    [
        pytest.param(LomapAtomMapper, "time", 30, id="Lomap"),
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


def test_fec_full_workflow(tyk2_ligands, tyk2_protein):
    """Make sure we can run the full FEC workflow"""
    factory = FreeEnergyCalculationFactory()
    # change the default settings to make sure they propagated
    # change the lomap timeout
    factory.network_planner.atom_mapping_engine.time = 30
    factory.simulation_settings.equilibration_length = 0.5 * OFFUnit.nanoseconds
    # plan a network
    planned_network = factory.create_fec_dataset(
        dataset_name="TYK2-test-dataset", receptor=tyk2_protein, ligands=tyk2_ligands
    )
    # make sure the settings were used correctly
    assert planned_network.network.atom_mapping_engine.time == 30
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
