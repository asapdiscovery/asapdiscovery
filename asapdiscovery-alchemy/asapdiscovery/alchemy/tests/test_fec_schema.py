import functools

import openfe
import pytest
from alchemiscale import Scope, ScopedKey
from asapdiscovery.alchemy.schema.atom_mapping import (
    KartografAtomMapper,
    LomapAtomMapper,
    PersesAtomMapper,
)
from asapdiscovery.alchemy.schema.fec import (
    AdaptiveSettings,
    AlchemiscaleResults,
    FreeEnergyCalculationFactory,
    SolventSettings,
    TransformationResult,
)
from asapdiscovery.alchemy.schema.network import (
    CustomNetworkPlanner,
    MaximalPlanner,
    MinimalRedundantPlanner,
    MinimalSpanningPlanner,
    NetworkPlanner,
    RadialPlanner,
)
from asapdiscovery.alchemy.utils import extract_custom_ligand_network
from asapdiscovery.data.schema.identifiers import BespokeParameter, BespokeParameters
from openff.toolkit import ForceField
from openff.units import unit as OFFUnit


@pytest.mark.parametrize(
    "mapper, argument, value",
    [
        pytest.param(LomapAtomMapper, "max3d", 30, id="Lomap"),
        pytest.param(
            PersesAtomMapper,
            "coordinate_tolerance",
            0.15,
            id="Perses",
            marks=pytest.mark.xfail(reason="upstream OpenFE #929"),
        ),
        pytest.param(
            KartografAtomMapper, "map_exact_ring_matches_only", True, id="Kartograph"
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
    "network_planner, openfe_func",
    [
        pytest.param(RadialPlanner, "radial", id="Radial"),
        pytest.param(MaximalPlanner, "maximal", id="Maximal"),
        pytest.param(MinimalSpanningPlanner, "minimal_spanning", id="Minimal Spanning"),
        pytest.param(
            MinimalRedundantPlanner, "minimal_redundant_network", id="Minimal redundant"
        ),
    ],
)
def test_network_planner_get_network(network_planner, openfe_func):
    """Make sure we get the correct network planner based on the network_planning_method setting."""

    planner = network_planner()

    planning_func = planner.get_planning_function()
    # check the name of the callable, special case for functools wrapped
    if isinstance(planning_func, functools.partial):
        assert openfe_func in planning_func.func.__name__
    else:
        assert openfe_func in planning_func.__name__


def test_plan_from_names(tyk2_ligands, tyk2_small_custom_network):
    """Make sure we can plan a network using the names of the ligands."""
    edges = extract_custom_ligand_network(tyk2_small_custom_network)
    planner = NetworkPlanner(network_planning_method=CustomNetworkPlanner(edges=edges))
    network = planner.generate_network(ligands=tyk2_ligands).to_ligand_network()
    # make sure the edges are as we expect
    for edge in network.edges:
        assert (edge.componentA.name, edge.componentB.name) in edges


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
        pytest.param(RadialPlanner, id="Radial"),
        pytest.param(MaximalPlanner, id="Maximal"),
        pytest.param(MinimalSpanningPlanner, id="Minimal Spanning"),
        pytest.param(MinimalRedundantPlanner, id="Minimal redundant"),
    ],
)
def test_generate_network_lomap(network_type, tyk2_ligands):
    """Test generating ligand FEC networks with the configured settings using lomap."""

    network_planning_method = network_type()
    if network_planning_method.type == "RadialPlanner":
        central = tyk2_ligands[0]
        ligands = tyk2_ligands[1:]
    else:
        central = None
        ligands = tyk2_ligands
    # configure the mapper
    planner = NetworkPlanner(
        atom_mapping_engine=LomapAtomMapper(),
        scorer="default_lomap",
        network_planning_method=network_planning_method,
    )

    planned_network = planner.generate_network(ligands=ligands, central_ligand=central)

    fe_network = planned_network.to_ligand_network()
    # make sure we have all the ligands we expect
    assert len(fe_network.nodes) == 10
    if network_planning_method.type == "RadialPlanner":
        # radial should have all ligands connected to the central node
        assert len(fe_network.edges) == 9

    elif network_planning_method.type == "MinimalSpanningPlanner":
        # there should be only 1 edge connecting each ligand to the network
        assert len(fe_network.edges) == 9

    elif network_planning_method.type == "MinimalSpanningPlanner":
        # there should be two minimal networks
        assert len(fe_network.edges) == 18

    # make sure we can convert back to openfe ligands
    openfe_ligands = planned_network.to_openfe_ligands()
    assert len(openfe_ligands) == 10
    assert isinstance(openfe_ligands[0], openfe.SmallMoleculeComponent)


def test_plan_radial_error(tyk2_ligands):
    """Make sure an error is raised if we try and plan a radial network with no central ligand"""
    planner = NetworkPlanner(network_planning_method=RadialPlanner())
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


def test_fec_adaptive_sampling(tyk2_ligands, tyk2_protein):
    """Make sure we can adjust simulation settings in the factory adaptively"""
    tyk2_ligands_mini = tyk2_ligands[:3]

    # define some adaptive settings for sampling time and the final sampling
    # times this should result in. This assumes default_lomap_scorer as network planner
    adaptive_settings = AdaptiveSettings(
        adaptive_sampling=True,
        adaptive_sampling_multiplier=2,
        adaptive_sampling_threshold=0.9,
    )
    factory = FreeEnergyCalculationFactory(adaptive_settings=adaptive_settings)

    reference_adaptive_sampling_times = [5.0, 10.0, 5.0, 5.0, 10.0, 5.0]

    # create an alchemicalnetwork with these settings and test that the adaptive
    # settings were applied
    alchemical_network = factory.create_fec_dataset(
        dataset_name="TYK2-test-dataset-duplicated",
        receptor=tyk2_protein,
        ligands=tyk2_ligands_mini,
    ).to_alchemical_network()
    sampling_lengths = []
    for edge in alchemical_network.edges:
        sampling_lengths.append(
            edge.protocol.settings.simulation_settings.production_length.magnitude
        )
    # test while sorting because edge order is scrambled randomly
    assert sorted(reference_adaptive_sampling_times) == sorted(sampling_lengths)


def test_fec_adaptive_sampling_disabled(tyk2_ligands, tyk2_protein):
    """Make sure we can disable adaptive simulation settings in the factory"""
    tyk2_ligands_mini = tyk2_ligands[:3]

    # now repeat but with disabling adaptive settings. This should make the sampling time
    # the same for all edges.
    adaptive_settings = AdaptiveSettings(
        adaptive_sampling=False,
    )
    factory = FreeEnergyCalculationFactory(adaptive_settings=adaptive_settings)
    reference_adaptive_sampling_times = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    # Test that the adaptive settings were NOT applied
    alchemical_network = factory.create_fec_dataset(
        dataset_name="TYK2-test-dataset-duplicated",
        receptor=tyk2_protein,
        ligands=tyk2_ligands_mini,
    ).to_alchemical_network()
    sampling_lengths = []
    for edge in alchemical_network.edges:
        sampling_lengths.append(
            edge.protocol.settings.simulation_settings.production_length.magnitude
        )
    assert reference_adaptive_sampling_times == sampling_lengths


def test_fec_adaptive_solvent_padding(tyk2_ligands, tyk2_protein):
    """Make sure we can adjust simulation settings in the factory adaptively"""
    tyk2_ligands_mini = tyk2_ligands[:3]

    # define some adaptive settings for solvent padding
    adaptive_settings = AdaptiveSettings(
        adaptive_solvent_padding=True,
        solvent_padding_complex=2 * OFFUnit.nanometer,
        solvent_padding_solvated=3 * OFFUnit.nanometer,
    )
    factory = FreeEnergyCalculationFactory(adaptive_settings=adaptive_settings)
    reference_adaptive_padding_sizes = [2.0, 3.0, 2.0, 3.0, 3.0, 2.0]

    # create an alchemicalnetwork with these settings and test that the adaptive
    # settings were applied
    alchemical_network = factory.create_fec_dataset(
        dataset_name="TYK2-test-dataset-duplicated",
        receptor=tyk2_protein,
        ligands=tyk2_ligands_mini,
    ).to_alchemical_network()
    padding_sizes = []
    for edge in alchemical_network.edges:
        padding_sizes.append(
            edge.protocol.settings.solvation_settings.solvent_padding.magnitude
        )
    # test while sorting because edge/phase order is scrambled randomly
    assert sorted(reference_adaptive_padding_sizes) == sorted(padding_sizes)


def test_fec_adaptive_solvent_padding_disabled(tyk2_ligands, tyk2_protein):
    """Make sure we can adjust simulation settings in the factory adaptively"""
    tyk2_ligands_mini = tyk2_ligands[:3]

    # Repeat but now without adaptive solvent padding sizes. This will pass
    # for any solvent padding size as long as it's the same across all edges/phases
    adaptive_settings = AdaptiveSettings(adaptive_solvent_padding=False)
    factory = FreeEnergyCalculationFactory(adaptive_settings=adaptive_settings)

    # create an alchemicalnetwork with these settings and test that the adaptive
    # settings were NOT applied
    alchemical_network = factory.create_fec_dataset(
        dataset_name="TYK2-test-dataset-duplicated",
        receptor=tyk2_protein,
        ligands=tyk2_ligands_mini,
    ).to_alchemical_network()
    padding_sizes = []
    for edge in alchemical_network.edges:
        padding_sizes.append(
            edge.protocol.settings.solvation_settings.solvent_padding.magnitude
        )
    # test that only one padding size is defined across all edges/phases
    assert len(set(padding_sizes)) == 1


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


def test_fec_dataset_missing_names(tyk2_ligands, tyk2_protein):
    """Make sure missing ligand names are caught"""
    ligands = [ligand.copy(deep=True) for ligand in tyk2_ligands]
    ligands[0].compound_name = ""

    factory = FreeEnergyCalculationFactory()
    with pytest.raises(
        ValueError, match=f"1 of {len(ligands)} ligands do not have names"
    ):
        _ = factory.create_fec_dataset(
            dataset_name="TYK2-test-dataset-missing-name",
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


def test_fec_with_bespoke_parameters(tyk2_fec_network):
    """
    Make sure we can generate an OpenFE network with bespoke torsions parameters added
    to the force field.
    """
    # mock some torsion parameters which hit all tyk2 amide torsions.
    bespoke_parameters = BespokeParameters(
        base_force_field=tyk2_fec_network.forcefield_settings.small_molecule_forcefield
    )
    bespoke_parameters.parameters.append(
        BespokeParameter(
            interaction="ProperTorsions",
            smirks="[#17]-[#6a:1]@[#6a:2]-[#6:3]=[#8:4]",
            values={"k1": 10, "k2": 1, "k3": 2, "k4": 3},
        )
    )
    # add this parameter to each of the ligands
    for ligand in tyk2_fec_network.network.ligands:
        ligand.bespoke_parameters = bespoke_parameters
    # trigger injecting the bespoke parameters
    alchem_network = tyk2_fec_network.to_alchemical_network()
    # check each edge has the bespoke parameters added to the force field
    for transformation in alchem_network.edges:
        off = ForceField(
            transformation.protocol.settings.forcefield_settings.small_molecule_forcefield
        )
        bespoke_param = bespoke_parameters.parameters[0]
        handler = off.get_parameter_handler(bespoke_param.interaction)
        off_param = handler[bespoke_param.smirks]
        assert off_param.k1.m == 10.0


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
    with pytest.warns(
        UserWarning,
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


def test_results_to_cinnabar_with_prediction(tyk2_result_network):
    """Make sure we can predict the absolute and relative free energies using cinnabar"""

    fe_map = tyk2_result_network.results.to_fe_map()
    # generate the absolute values using MLE
    fe_map.generate_absolute_values()
    # check we get the expected results
    absolute_dataframe = fe_map.get_absolute_dataframe()
    # define some rows to check
    row_refs = [
        (0, "lig_ejm_31", -0.133223, 0.075722),
        (2, "lig_ejm_42", 0.678041, 0.093269),
        (4, "lig_ejm_48", 0.771667, 0.314375),
    ]
    # grab and check the row calculated row
    for row, label, dg, uncertainty in row_refs:
        lig_row = absolute_dataframe.iloc[row]
        assert lig_row["label"] == label
        assert lig_row["DG (kcal/mol)"] == pytest.approx(dg, abs=1e-6)
        assert lig_row["uncertainty (kcal/mol)"] == pytest.approx(uncertainty, abs=1e-6)
        assert lig_row["source"] == "MLE"
        assert lig_row["computational"]

    relative_dataframe = fe_map.get_relative_dataframe()
    # define some rows to check in the relative dataframe
    row_refs_rel = [
        (0, "lig_ejm_31", "lig_ejm_47", 0.111551, 0.149755),
        (1, "lig_ejm_31", "lig_ejm_42", 0.811265, 0.0861),
        (4, "lig_ejm_42", "lig_ejm_50", 0.172555, 0.166535),
    ]
    for row, label_a, label_b, ddg, uncertainty in row_refs_rel:
        relative_row = relative_dataframe.iloc[row]
        assert relative_row["labelA"] == label_a
        assert relative_row["labelB"] == label_b
        assert relative_row["DDG (kcal/mol)"] == pytest.approx(ddg, abs=1e-6)
        assert relative_row["uncertainty (kcal/mol)"] == pytest.approx(
            uncertainty, abs=1e-6
        )
        assert relative_row["source"] == "calculated"
