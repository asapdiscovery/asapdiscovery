"""Tests for membrane system building.

The geometry tests use a synthetic four-helix bundle rather than a real structure, so they
run anywhere and exercise the code paths that matter: axis detection, midplane placement,
landmark validation and the in-plane rotation search.
"""

import numpy as np
import pytest

from asapdiscovery.alchemy.membrane import (
    HERG,
    MembraneAlignmentSpec,
    MembraneBuildSettings,
    ResidueRange,
    StructuralIonSite,
    best_inplane_angle,
    check_landmarks,
    get_preset,
    membrane_axis,
    membrane_placement,
    read_pdb_atoms,
    rotation_to_z,
    structural_ion_positions,
)


def _atom(chain, resi, resn, name, xyz, el="C"):
    return {
        "chain": chain,
        "resi": resi,
        "resn": resn,
        "name": name,
        "el": el,
        "xyz": np.array(xyz, dtype=float),
    }


@pytest.fixture
def toy_channel():
    """A four-fold symmetric bundle with its axis along +x, so alignment has work to do.

    Residues 1-20 are the 'TM helix' on each of four chains, arranged around the axis;
    residue 30 marks the 'extracellular' end and residue 40 the 'intracellular' end.
    """
    atoms = []
    for chain, (dy, dz) in zip("ABCD", [(10, 0), (0, 10), (-10, 0), (0, -10)]):
        # TM helix: spans x from -13 to +13, i.e. 26 A of 'membrane'
        for i, resi in enumerate(range(1, 21)):
            x = -13.0 + 26.0 * i / 19.0
            atoms.append(_atom(chain, resi, "LEU", "CA", (x, dy, dz)))
            # a sidechain heavy atom further out, for the hydrophobic cross-check
            atoms.append(_atom(chain, resi, "LEU", "CD1", (x, dy * 1.4, dz * 1.4)))
        # axis landmarks, on the axis by symmetry
        atoms.append(_atom(chain, 30, "GLY", "CA", (20.0, dy * 0.1, dz * 0.1)))
        atoms.append(_atom(chain, 40, "GLY", "CA", (-20.0, dy * 0.1, dz * 0.1)))
    return atoms


@pytest.fixture
def toy_spec():
    return MembraneAlignmentSpec(
        name="toy",
        axis_from=ResidueRange(start=30, end=30),
        axis_to=ResidueRange(start=40, end=40),
        tm_helices={"H1": ResidueRange(start=1, end=20)},
        midplane_helices=["H1"],
        landmarks={"tip": ResidueRange(start=30, end=30)},
    )


def test_rotation_to_z_maps_axis_onto_z():
    for axis in ([1, 0, 0], [0, 1, 0], [1, 1, 1], [-0.14, 0.019, -0.99]):
        rot = rotation_to_z(axis)
        mapped = rot @ (np.array(axis, float) / np.linalg.norm(axis))
        assert np.allclose(mapped, [0, 0, 1], atol=1e-9)


def test_rotation_to_z_handles_degenerate_cases():
    assert np.allclose(rotation_to_z([0, 0, 1]), np.eye(3))
    # antiparallel: must still map onto +z
    assert np.allclose(
        rotation_to_z([0, 0, -1]) @ np.array([0.0, 0.0, -1.0]), [0, 0, 1]
    )


def test_membrane_axis_finds_symmetry_axis(toy_channel, toy_spec):
    axis, normal, start, end = membrane_axis(toy_channel, toy_spec)
    # axis runs axis_from (x=+20) -> axis_to (x=-20), so it points along -x
    assert np.allclose(axis, [-1, 0, 0], atol=1e-6)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    # the chain-centroid normal is a genuinely independent estimate of the same axis
    assert abs(abs(float(normal @ axis)) - 1.0) < 0.05
    assert start[0] > end[0]


def test_membrane_axis_reports_missing_landmarks(toy_channel):
    spec = MembraneAlignmentSpec(
        name="wrong-numbering",
        axis_from=ResidueRange(start=900, end=910),
        axis_to=ResidueRange(start=40, end=40),
        tm_helices={"H1": ResidueRange(start=1, end=20)},
        midplane_helices=["H1"],
    )
    with pytest.raises(ValueError, match="axis landmarks not found"):
        membrane_axis(toy_channel, spec)


def test_membrane_placement_centres_on_helix_span(toy_channel, toy_spec):
    axis, _, start, _ = membrane_axis(toy_channel, toy_spec)
    rot = rotation_to_z(-axis)
    placement = membrane_placement(toy_channel, rot, start, toy_spec)
    # the helix spans 26 A and is symmetric about the bundle centre
    assert placement["tm_helix_spans"]["H1"]["span"] == pytest.approx(26.0, abs=0.5)
    # re-centre on the midplane and confirm it lands at zero
    origin = start + np.linalg.inv(rot) @ np.array(
        [0, 0, placement["membrane_centre_z"]]
    )
    recentred = membrane_placement(toy_channel, rot, origin, toy_spec)
    assert recentred["membrane_centre_z"] == pytest.approx(0.0, abs=1e-6)


def test_membrane_placement_rejects_unknown_helices(toy_channel, toy_spec):
    spec = toy_spec.model_copy(
        update={"tm_helices": {"H1": ResidueRange(start=800, end=820)}}
    )
    axis, _, start, _ = membrane_axis(toy_channel, toy_spec)
    with pytest.raises(ValueError, match="none of the midplane helices"):
        membrane_placement(toy_channel, rotation_to_z(-axis), start, spec)


def test_check_landmarks_flags_out_of_bounds():
    placement = {"landmark_z_after_centring": {"Tyr652": 25.0, "Phe656": -3.0}}
    spec = HERG
    violations = check_landmarks(placement, spec)
    # Tyr652 at +25 A is far outside its (-8, 8) window; Phe656 is fine
    assert len(violations) == 1
    assert "Tyr652" in violations[0]
    assert "+25.0" in violations[0]


def test_check_landmarks_passes_on_measured_herg_values():
    """The depths actually measured for 8ZYP and 8ZYO must satisfy the shipped preset."""
    for measured in (
        {  # 8ZYP
            "turret_582_597": 30.7,
            "selectivity_filter_624_628": 15.3,
            "Tyr652": 1.0,
            "Phe656": -2.7,
            "S6_end_658_665": -9.8,
        },
        {  # 8ZYO
            "turret_582_597": 30.1,
            "selectivity_filter_624_628": 14.6,
            "Tyr652": 0.2,
            "Phe656": -3.8,
            "S6_end_658_665": -11.1,
        },
    ):
        assert check_landmarks({"landmark_z_after_centring": measured}, HERG) == []


def test_best_inplane_angle_prefers_smaller_footprint():
    """A square footprint rotated 45 deg needs a bigger axis-aligned box, not a smaller one."""
    # a diamond: extent is minimal at 45 deg, maximal at 0
    diamond = np.array([[3.0, 0, 0], [-3.0, 0, 0], [0, 3.0, 0], [0, -3.0, 0]])
    _, degrees, (nx, ny), extent = best_inplane_angle(diamond, (4.0, 4.0), 0.5)
    assert nx >= 1 and ny >= 1
    # at the chosen angle the box must be no larger than at zero rotation
    at_zero = np.ceil((6.0 + 1.0) / 4.0) ** 2
    assert nx * ny <= at_zero


def test_structural_ion_positions_sits_between_carbonyl_rings():
    atoms = []
    for chain, (dx, dy) in zip("ABCD", [(3, 0), (0, 3), (-3, 0), (0, -3)]):
        atoms.append(_atom(chain, 624, "SER", "O", (dx, dy, 0.0), el="O"))
        atoms.append(_atom(chain, 625, "VAL", "O", (dx, dy, 3.0), el="O"))
    spec = MembraneAlignmentSpec(
        name="cage",
        axis_from=ResidueRange(start=624, end=624),
        axis_to=ResidueRange(start=625, end=625),
        tm_helices={"H1": ResidueRange(start=624, end=625)},
        midplane_helices=["H1"],
        structural_ions=[
            StructuralIonSite(
                name="S4", element="K", lower_residue=624, upper_residue=625
            )
        ],
    )
    ions = structural_ion_positions(atoms, np.eye(3), np.zeros(3), spec)
    assert len(ions) == 1
    ion = ions[0]
    # midway between the two rings, on the axis
    assert ion["xyz"][2] == pytest.approx(1.5)
    assert ion["radial_offset_A"] == pytest.approx(0.0, abs=1e-6)
    # all eight carbonyls coordinate it
    assert ion["n_coord_O"] == 8


def test_structural_ion_positions_skips_missing_cage(toy_channel, toy_spec):
    spec = toy_spec.model_copy(
        update={
            "structural_ions": [
                StructuralIonSite(name="X", lower_residue=900, upper_residue=901)
            ]
        }
    )
    assert structural_ion_positions(toy_channel, np.eye(3), np.zeros(3), spec) == []


def test_herg_preset_is_self_consistent():
    spec = get_preset("herg")
    assert spec is HERG
    assert spec.midplane_helices == ["S1", "S2"]
    # S4 is measured but must NOT set the midplane: it is the voltage sensor, and its
    # gating arginines sit inside the bilayer by design
    assert "S4" in spec.tm_helices
    assert "S4" not in spec.midplane_helices
    # every bounded landmark is a defined landmark (also enforced by the validator)
    assert set(spec.landmark_bounds) <= set(spec.landmarks)


def test_get_preset_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown membrane preset"):
        get_preset("not-a-real-target")


def test_alignment_spec_rejects_undefined_midplane_helix():
    with pytest.raises(ValueError, match="not defined in tm_helices"):
        MembraneAlignmentSpec(
            name="bad",
            axis_from=ResidueRange(start=1, end=2),
            axis_to=ResidueRange(start=3, end=4),
            tm_helices={"H1": ResidueRange(start=1, end=10)},
            midplane_helices=["H2"],
        )


def test_alignment_spec_rejects_inverted_residue_range():
    with pytest.raises(ValueError, match="is before start"):
        ResidueRange(start=100, end=50)


def test_alignment_spec_round_trips_through_json(tmpdir):
    with tmpdir.as_cwd():
        HERG.to_file("spec.json")
        assert MembraneAlignmentSpec.from_file("spec.json") == HERG


def test_build_settings_lipid_resnames_cover_pdb_truncation():
    """PDB truncates residue names to three characters, so both spellings must match.

    Anything selecting lipids that accepts only one silently matches zero, which is how a
    build once reported leaflets 0/0 and a protein core area of 205 nm^2.
    """
    settings = MembraneBuildSettings()
    assert settings.lipid_resnames == ("POPC", "POP")
    assert MembraneBuildSettings(lipid="DPP").lipid_resnames == ("DPP",)


def test_build_settings_default_forcefield_uses_amber14_lipid17():
    """`amber/lipid17.xml` also loads but supplies NO lipid template; amber14 is required."""
    assert "amber14/lipid17.xml" in MembraneBuildSettings().forcefields


def test_read_pdb_atoms_parses_records(tmpdir):
    pdb = (
        "ATOM      1  CA  LEU A 407      11.104   6.134  -6.504  1.00  0.00           C\n"
        "HETATM    2  P   POP B   1       1.000   2.000   3.000  1.00  0.00           P\n"
        "TER\n"
    )
    path = tmpdir.join("t.pdb")
    path.write(pdb)
    atoms = read_pdb_atoms(str(path))
    assert len(atoms) == 2
    assert atoms[0]["chain"] == "A" and atoms[0]["resi"] == 407
    assert (
        atoms[0]["resn"] == "LEU" and atoms[0]["name"] == "CA" and atoms[0]["el"] == "C"
    )
    assert np.allclose(atoms[0]["xyz"], [11.104, 6.134, -6.504])
    assert atoms[1]["resn"] == "POP" and atoms[1]["el"] == "P"


# --------------------------------------------------------------------------------------
# Membrane support in the FEC schema.
#
# These need gufe's ProteinMembraneComponent, which landed in openfe 1.11 -- i.e. the
# `alchemy-membrane` environment. They are skipped on the openfe 1.8 stack.
# --------------------------------------------------------------------------------------

import openfe  # noqa: E402

from asapdiscovery.alchemy.schema._util import (  # noqa: E402
    is_membrane_receptor,
    is_presolvated_receptor,
)
from asapdiscovery.alchemy.schema.protocols import (  # noqa: E402
    LIPID_FORCEFIELD,
    apply_membrane_settings,
    available_protocols,
    default_protocol_settings,
    supports_membrane,
)

requires_membrane_openfe = pytest.mark.skipif(
    not hasattr(openfe, "ProteinMembraneComponent"),
    reason=(
        f"openfe {openfe.__version__} has no ProteinMembraneComponent; membrane support "
        "needs openfe >=1.11 (the 'alchemy-membrane' environment)"
    ),
)


@pytest.fixture(scope="session")
def membrane_receptor(tmp_path_factory):
    """A ProteinMembraneComponent built from a solvated protein box.

    No actual lipids: gufe's membrane distinction is conveyed *solely through the
    component type* ("This type serves as a semantic marker enabling membrane-specific
    simulation behavior"), so a solvated box with a box vector is enough to exercise every
    ASAP-Alchemy code path that branches on it. The lipid physics is openfe's to test.
    """
    pytest.importorskip("openmm")
    if not hasattr(openfe, "ProteinMembraneComponent"):
        pytest.skip("openfe too old for ProteinMembraneComponent")

    from openmm import app, unit

    from asapdiscovery.data.testing.test_resources import fetch_test_file

    pdb = app.PDBFile(fetch_test_file("tyk2_protein.pdb").as_posix())
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        forcefield, padding=0.9 * unit.nanometer, ionicStrength=0.15 * unit.molar
    )
    out = tmp_path_factory.mktemp("membrane") / "solvated.pdb"
    with open(out, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)

    component = openfe.ProteinMembraneComponent.from_pdb_file(
        out.as_posix(), name="toy_membrane"
    )
    component.validate()
    return component


def test_supports_membrane_registry():
    """RFE, SepTop and ABFE handle membranes in openfe 1.12; the F@H protocols do not."""
    assert supports_membrane("RelativeHybridTopologyProtocol")
    assert supports_membrane("SepTopProtocol")
    assert supports_membrane("AbsoluteBindingProtocol")
    # the F@H compute services run a pinned, older openfe
    assert not supports_membrane("FahNonEquilibriumCyclingProtocol")
    assert not supports_membrane("NonEquilibriumCyclingProtocol")


def test_supports_membrane_rejects_unknown_protocol():
    with pytest.raises(KeyError, match="Unknown protocol"):
        supports_membrane("NotAProtocol")


def test_apply_membrane_settings_rejects_unsupported_protocol():
    settings = default_protocol_settings("RelativeHybridTopologyProtocol")
    with pytest.raises(NotImplementedError, match="does not support membrane"):
        apply_membrane_settings("NonEquilibriumCyclingProtocol", settings)


@requires_membrane_openfe
def test_apply_membrane_settings_sets_membrane_barostat():
    settings = default_protocol_settings(
        "RelativeHybridTopologyProtocol"
    ).unfrozen_copy()
    assert settings.integrator_settings.barostat == "MonteCarloBarostat"

    apply_membrane_settings("RelativeHybridTopologyProtocol", settings)

    assert settings.integrator_settings.barostat == "MonteCarloMembraneBarostat"
    assert LIPID_FORCEFIELD in settings.forcefield_settings.forcefields


@requires_membrane_openfe
def test_apply_membrane_settings_adds_lipid_forcefield_when_customised():
    """openfe >=1.12 ships the lipid force field by default, so the append only bites when
    the caller has supplied their own force field list -- which is exactly what a tuned
    production factory does, and how a bilayer ends up with no lipid parameters."""
    settings = default_protocol_settings(
        "RelativeHybridTopologyProtocol"
    ).unfrozen_copy()
    settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",
        "amber/tip3p_standard.xml",
    ]
    apply_membrane_settings("RelativeHybridTopologyProtocol", settings)
    assert LIPID_FORCEFIELD in settings.forcefield_settings.forcefields


@requires_membrane_openfe
def test_apply_membrane_settings_is_idempotent():
    """Applying twice must not append the lipid force field twice."""
    settings = default_protocol_settings(
        "RelativeHybridTopologyProtocol"
    ).unfrozen_copy()
    apply_membrane_settings("RelativeHybridTopologyProtocol", settings)
    apply_membrane_settings("RelativeHybridTopologyProtocol", settings)
    assert settings.forcefield_settings.forcefields.count(LIPID_FORCEFIELD) == 1


@requires_membrane_openfe
def test_apply_membrane_settings_targets_complex_leg_for_septop():
    """SepTop splits its integrator settings per leg; only the complex one is a membrane."""
    settings = default_protocol_settings("SepTopProtocol").unfrozen_copy()
    apply_membrane_settings("SepTopProtocol", settings)
    assert settings.complex_integrator_settings.barostat == "MonteCarloMembraneBarostat"
    assert settings.solvent_integrator_settings.barostat == "MonteCarloBarostat"


@requires_membrane_openfe
def test_receptor_predicates(membrane_receptor, tyk2_protein):
    assert is_membrane_receptor(membrane_receptor)
    assert is_presolvated_receptor(membrane_receptor)
    # a plain soluble receptor is neither
    assert not is_membrane_receptor(tyk2_protein)
    assert not is_presolvated_receptor(tyk2_protein)


@requires_membrane_openfe
def test_network_round_trips_membrane_receptor(membrane_receptor, tyk2_ligands):
    """The receptor must come back as a ProteinMembraneComponent, not a ProteinComponent.

    The component *type* is what selects the membrane barostat downstream, so a receptor
    that silently degrades to ProteinComponent on deserialisation would run the bilayer
    under an isotropic barostat.
    """
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationFactory

    factory = FreeEnergyCalculationFactory()
    planned = factory.create_fec_dataset(
        dataset_name="membrane_round_trip",
        receptor=membrane_receptor,
        ligands=tyk2_ligands[:3],
    )
    assert planned.is_membrane
    restored = planned.to_openfe_receptor()
    assert isinstance(restored, openfe.ProteinMembraneComponent)
    assert restored == membrane_receptor


@requires_membrane_openfe
def test_membrane_alchemical_network_composition(membrane_receptor, tyk2_ligands):
    """The load-bearing assertions: no solvent on the complex leg, membrane barostat on it."""
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationFactory

    factory = FreeEnergyCalculationFactory()
    planned = factory.create_fec_dataset(
        dataset_name="membrane_network",
        receptor=membrane_receptor,
        ligands=tyk2_ligands[:3],
    )
    network = planned.to_alchemical_network()

    complex_edges = [e for e in network.edges if e.stateA.name.endswith("_complex")]
    solvent_edges = [e for e in network.edges if e.stateA.name.endswith("_solvent")]
    assert complex_edges and solvent_edges

    for edge in complex_edges:
        components = edge.stateA.components
        # the membrane PDB supplies its own water, ions and box
        assert (
            "solvent" not in components
        ), "complex leg must not carry a SolventComponent"
        assert isinstance(components["protein"], openfe.ProteinMembraneComponent)
        integrator = edge.protocol.settings.integrator_settings
        assert integrator.barostat == "MonteCarloMembraneBarostat"
        assert (
            LIPID_FORCEFIELD in edge.protocol.settings.forcefield_settings.forcefields
        )

    for edge in solvent_edges:
        components = edge.stateA.components
        assert "solvent" in components, "solvent leg still needs a SolventComponent"
        assert "protein" not in components
        # the solvent leg is an ordinary water box, so it keeps the isotropic barostat.
        # Note the lipid force field is NOT asserted absent: openfe >=1.12 ships it in
        # the defaults, so its presence here says nothing either way.
        assert (
            edge.protocol.settings.integrator_settings.barostat == "MonteCarloBarostat"
        )


@requires_membrane_openfe
def test_soluble_receptor_is_untouched(tyk2_protein, tyk2_ligands):
    """A regression guard: the non-membrane path must not acquire a membrane barostat."""
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationFactory

    factory = FreeEnergyCalculationFactory()
    planned = factory.create_fec_dataset(
        dataset_name="soluble_network",
        receptor=tyk2_protein,
        ligands=tyk2_ligands[:3],
    )
    assert not planned.is_membrane
    network = planned.to_alchemical_network()
    for edge in network.edges:
        assert "solvent" in edge.stateA.components
        assert (
            edge.protocol.settings.integrator_settings.barostat == "MonteCarloBarostat"
        )


@requires_membrane_openfe
def test_membrane_network_rejects_incapable_protocol(membrane_receptor, tyk2_ligands):
    """Fail at planning time, not after 120 transformations have been submitted."""
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationFactory

    incapable = [p for p in available_protocols() if not supports_membrane(p)]
    factory = FreeEnergyCalculationFactory(protocol=[incapable[0]])
    planned = factory.create_fec_dataset(
        dataset_name="bad_membrane_network",
        receptor=membrane_receptor,
        ligands=tyk2_ligands[:3],
    )
    with pytest.raises(NotImplementedError, match="cannot run membrane systems"):
        planned.to_alchemical_network()
