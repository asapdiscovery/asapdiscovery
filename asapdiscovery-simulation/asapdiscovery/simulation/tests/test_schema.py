import openfe
import pytest
from rdkit import Chem

from asapdiscovery.simulation.schema import (
    KartographAtomMapper,
    LomapAtomMapper,
    NetworkPlanner,
    PersesAtomMapper,
    SolventSettings,
)


@pytest.mark.parametrize(
    "mapper, argument, value",
    [
        pytest.param(LomapAtomMapper, "time", 30, id="Lomap"),
        pytest.param(PersesAtomMapper, "coordinate_tolerance", 0.15, id="Perses"),
        pytest.param(
            KartographAtomMapper, "atom_ring_matches_ring", True, id="Kartograph"
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
            KartographAtomMapper, ["openfe", "rdkit", "kartograf"], id="Kartograph"
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
def test_generate_network_lomap(network_type):
    """Test generating ligand FEP networks with the configured settings using lomap."""

    supp = Chem.SDMolSupplier("data/tyk2_ligands.sdf", removeHs=False)
    # convert to openfe objects
    ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]
    if network_type == "radial":
        central = ligands.pop(0)
    else:
        central = None
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


def test_solvent_settings():
    """Make sure solvent settings are correctly passed to the gufe solvent component."""

    settings = SolventSettings()
    settings.ion_concentration = 0.25

    component = settings.to_solvent_component()
    assert component._ion_concentration.m == settings.ion_concentration
