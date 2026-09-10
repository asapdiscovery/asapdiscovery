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
    align_receptor,
    best_inplane_angle,
    check_landmarks,
    composition,
    diagnostics,
    get_preset,
    lipid_patch,
    membrane_axis,
    membrane_placement,
    read_pdb_atoms,
    rotation_to_z,
    structural_ion_positions,
    wrap_molecules,
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


# --------------------------------------------------------------------------------------
# builder: the OpenMM-side helpers.
#
# These need openmm but no GPU and no equilibration, so they cover the parts of
# builder.py that would otherwise only be exercised by a multi-hour production build.
# --------------------------------------------------------------------------------------

from asapdiscovery.alchemy.membrane.builder import (  # noqa: E402
    LIGAND_RESNAME,
    _non_protein_resnames,
)

openmm_app = pytest.importorskip("openmm.app", reason="openmm not installed")


def _topology_with(residues):
    """A minimal OpenMM Topology: one chain, one single-atom residue per name given."""
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain()
    for name in residues:
        residue = topology.addResidue(name, chain)
        topology.addAtom("C", app.element.carbon, residue)
    return topology


def test_lipid_patch_renames_truncated_residues():
    """PDB truncates POPC to POP; lipid17 defines POPC, so addMembrane needs the rename."""
    patch, n_renamed = lipid_patch(MembraneBuildSettings())
    assert n_renamed > 0
    names = {r.name for r in patch.topology.residues()}
    # the shipped patch is a *hydrated* bilayer, so water is expected alongside the lipid
    assert "POPC" in names
    assert "POP" not in names, f"unrenamed lipid residues remain: {names}"


def test_lipid_patch_rejects_lipid_openmm_does_not_ship():
    with pytest.raises(FileNotFoundError, match="no pre-equilibrated patch"):
        lipid_patch(MembraneBuildSettings(lipid="NOTALIPID"))


def test_composition_counts_each_category():
    topology = _topology_with(
        ["ALA", "GLY", "POPC", "POPC", "HOH", "HOH", "HOH", "NA", "CL", LIGAND_RESNAME]
    )
    counts = composition(topology, MembraneBuildSettings())
    assert counts["protein_residues"] == 2
    assert counts["lipids"] == 2
    assert counts["waters"] == 3
    assert counts["Na"] == 1 and counts["Cl"] == 1
    assert counts["ligand"] == 1
    assert counts["atoms"] == 10


def test_composition_accepts_truncated_lipid_name():
    """A system read back from a PDB carries POP, not POPC -- both must count as lipid."""
    counts = composition(_topology_with(["POP", "POP", "ALA"]), MembraneBuildSettings())
    assert counts["lipids"] == 2
    assert counts["protein_residues"] == 1


def test_composition_does_not_count_structural_ions_as_protein():
    """Regression: ion residue names come from the spec, so they cannot be hardcoded.

    A spec using Ca2+ would otherwise have its ions counted as protein residues -- and
    their atoms folded into the protein cross-section in diagnostics().
    """
    spec = HERG.model_copy(
        update={
            "structural_ions": [
                StructuralIonSite(
                    name="site", element="Ca", lower_residue=1, upper_residue=2
                )
            ]
        }
    )
    topology = _topology_with(["ALA", "CA", "CA"])
    without_spec = composition(topology, MembraneBuildSettings())
    with_spec = composition(topology, MembraneBuildSettings(), spec)
    assert without_spec["protein_residues"] == 3  # the bug this guards against
    assert with_spec["protein_residues"] == 1
    assert with_spec["structural_ions"] == {"CA": 2}


def test_non_protein_resnames_follows_the_spec():
    settings = MembraneBuildSettings()
    base = _non_protein_resnames(settings)
    assert {"HOH", "WAT", "NA", "CL", LIGAND_RESNAME, "POPC", "POP"} <= base
    assert "K" not in base
    # hERG declares potassium filter ions, so K becomes non-protein for that spec
    assert "K" in _non_protein_resnames(settings, HERG)


def test_wrap_molecules_moves_whole_molecules_only():
    """Per-atom wrapping splits molecules across the boundary; per-molecule does not.

    Getting this wrong once reported a leaflet split of 431/1 and a P-P thickness of
    27.1 A against a true ~37 A.
    """
    from openmm import app, unit

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    atoms = [
        topology.addAtom(n, app.element.oxygen, residue) for n in ("O", "H1", "H2")
    ]
    topology.addBond(atoms[0], atoms[1])
    topology.addBond(atoms[0], atoms[2])

    box = np.diag([4.0, 4.0, 4.0])
    # a molecule sitting a full box length out, straddling nothing
    positions = unit.Quantity(
        np.array([[4.4, 0.0, 0.0], [4.5, 0.0, 0.0], [4.6, 0.0, 0.0]]), unit.nanometer
    )
    wrapped = np.array(
        wrap_molecules(topology, positions, box, verbose=False).value_in_unit(
            unit.nanometer
        )
    )
    original = np.array(positions.value_in_unit(unit.nanometer))
    # the whole molecule moved by exactly one box vector -- internal geometry preserved
    shift = wrapped - original
    assert np.allclose(shift, shift[0]), "molecule was split across the boundary"
    assert np.allclose(shift[0], [-4.0, 0.0, 0.0])
    assert abs(wrapped[:, 0].mean()) <= 2.0


def test_wrap_molecules_leaves_centred_molecules_alone():
    from openmm import app, unit

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    positions = unit.Quantity(np.array([[0.1, -0.2, 0.3]]), unit.nanometer)
    wrapped = np.array(
        wrap_molecules(
            topology, positions, np.diag([4.0, 4.0, 4.0]), verbose=False
        ).value_in_unit(unit.nanometer)
    )
    assert np.allclose(wrapped, [[0.1, -0.2, 0.3]])


def test_diagnostics_splits_leaflets_and_measures_thickness():
    """A synthetic bilayer: phosphates at +-19 A give a 38 A P-P thickness."""
    from openmm import app, unit

    topology = app.Topology()
    chain = topology.addChain()
    coords = []
    for z in (19.0, -19.0):
        for i in range(8):
            residue = topology.addResidue("POPC", chain)
            topology.addAtom("P", app.element.phosphorus, residue)
            coords.append([float(i), 0.0, z])
    positions = unit.Quantity(np.array(coords), unit.angstrom)

    diag = diagnostics(
        topology, positions, np.diag([6.0, 6.0, 9.0]), MembraneBuildSettings()
    )
    assert diag["n_lipids_upper"] == 8
    assert diag["n_lipids_lower"] == 8
    assert diag["leaflet_imbalance"] == 0
    assert diag["P_P_thickness_A"] == pytest.approx(38.0, abs=0.1)
    assert diag["leaflet_split_z_A"] == pytest.approx(0.0, abs=1e-6)
    # 6x6 nm box over 8 lipids per leaflet
    assert diag["area_per_lipid_A2_uncorrected"] == pytest.approx(450.0, abs=1.0)


def test_diagnostics_handles_a_system_with_no_lipids():
    """Must not raise on an empty phosphate set -- min()/max() on an empty array would."""
    diag = diagnostics(
        _topology_with(["ALA"]),
        __import__("openmm").unit.Quantity(
            np.array([[0.0, 0.0, 0.0]]), __import__("openmm").unit.angstrom
        ),
        np.diag([4.0, 4.0, 4.0]),
        MembraneBuildSettings(),
    )
    assert diag["n_lipids_upper"] == 0 and diag["n_lipids_lower"] == 0
    assert diag["P_P_thickness_A"] is None
    assert diag["area_per_lipid_A2"] is None


def _write_pdb(path, atoms):
    """Write parsed-atom dicts back out as a minimal PDB."""
    with open(path, "w") as fh:
        for i, a in enumerate(atoms, 1):
            x, y, z = a["xyz"]
            fh.write(
                f"ATOM  {i:>5} {a['name']:<4}{a['resn']:>3} {a['chain']}"
                f"{a['resi']:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"  1.00  0.00          {a['el']:>2}\n"
            )
        fh.write("END\n")


def test_align_receptor_puts_the_axis_on_z(tmp_path, toy_channel, toy_spec):
    """End-to-end alignment: an axis along x must come out along z, midplane at zero."""
    pdb = tmp_path / "toy.pdb"
    _write_pdb(pdb, toy_channel)

    _, rot, origin, info = align_receptor(pdb.as_posix(), toy_spec)

    # the bundle axis was along x and is 90 degrees off z before alignment
    assert info["angle_axis_to_z_deg"] == pytest.approx(90.0, abs=0.5)
    # after alignment the midplane sits at z=0 by construction
    assert info["belt"]["membrane_centre_z"] == pytest.approx(0.0, abs=1e-6)
    # rot is a proper rotation: orthonormal, right-handed
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)
    # the extracellular landmark ends up at +z, not -z
    assert info["belt"]["landmark_z_after_centring"]["tip"] > 0
    assert info["landmark_violations"] == []
    # protein is centred in xy
    xy = np.array([rot @ (a["xyz"] - origin) for a in toy_channel])[:, :2]
    assert np.allclose(xy.mean(axis=0), [0.0, 0.0], atol=1e-6)


def test_align_receptor_strict_raises_but_non_strict_reports(tmp_path, toy_channel):
    """`analyse` needs the geometry of a BAD placement; `build` must refuse it."""
    impossible = MembraneAlignmentSpec(
        name="impossible",
        axis_from=ResidueRange(start=30, end=30),
        axis_to=ResidueRange(start=40, end=40),
        tm_helices={"H1": ResidueRange(start=1, end=20)},
        midplane_helices=["H1"],
        landmarks={"tip": ResidueRange(start=30, end=30)},
        # the tip really lands near +20 A, so this window cannot be satisfied
        landmark_bounds={"tip": (-100.0, -50.0)},
    )
    pdb = tmp_path / "toy.pdb"
    _write_pdb(pdb, toy_channel)

    with pytest.raises(ValueError, match="failed the landmark checks"):
        align_receptor(pdb.as_posix(), impossible)

    _, _, _, info = align_receptor(pdb.as_posix(), impossible, strict=False)
    assert len(info["landmark_violations"]) == 1
    assert "tip" in info["landmark_violations"][0]
    # ...and the geometry is still there to look at, which is the point
    assert "belt" in info and "inplane" in info


@pytest.mark.parametrize(
    "target,receptor,report",
    [
        pytest.param(
            "8ZYO",
            "posed_sets/_membrane/8ZYO_receptor_c4complete.pdb",
            "posed_sets/_membrane/8ZYO_membrane_report.json",
            id="8ZYO",
        ),
        pytest.param(
            "8ZYP",
            "posed_sets/cumming_8ZYP/prepared_receptor.pdb",
            "posed_sets/_membrane/8ZYP_membrane_report.json",
            id="8ZYP",
        ),
    ],
)
def test_align_receptor_reproduces_production_geometry(target, receptor, report):
    """Regression against the systems the hERG production networks were built on.

    Skipped unless the reference project is present; set ASAP_HERG_PROJECT to point at
    it. Guards the numbers that a change to the alignment maths would silently move.
    """
    import json
    import os
    import pathlib

    root = os.environ.get(
        "ASAP_HERG_PROJECT",
        os.path.expanduser(
            "~/projects/project_support/alchemy/2026-07-16_openadmet_hERG"
        ),
    )
    receptor_path = pathlib.Path(root) / receptor
    report_path = pathlib.Path(root) / report
    if not (receptor_path.exists() and report_path.exists()):
        pytest.skip(
            f"hERG reference data not found under {root}; set ASAP_HERG_PROJECT to run"
        )

    _, _, _, info = align_receptor(receptor_path.as_posix(), get_preset("herg"))
    reference = json.loads(report_path.read_text())

    assert info["membrane_axis"] == reference["pore_axis_deposited"]
    assert info["angle_axis_to_z_deg"] == reference["angle_axis_to_z_deg"]
    assert info["axis_vs_normal_deg"] == reference["axis_vs_normal_deg"]
    assert (
        info["protein_z_extent_after_align"]
        == reference["protein_z_extent_after_align"]
    )
    assert (
        info["inplane"]["inplane_spin_deg"] == reference["inplane"]["inplane_spin_deg"]
    )
    assert info["inplane"]["patch_copies_xy"] == reference["inplane"]["patch_copies_xy"]
    assert info["inplane"]["footprint_nm"] == reference["inplane"]["footprint_nm"]
    assert (
        info["belt"]["landmark_z_after_centring"]
        == reference["belt"]["landmark_z_after_centring"]
    )
    assert info["belt"]["tm_helix_spans"] == reference["belt"]["tm_helix_spans"]
    assert info["landmark_violations"] == []


# --------------------------------------------------------------------------------------
# is_membrane must not pay for deserialisation just to read a type name.
# --------------------------------------------------------------------------------------

from asapdiscovery.alchemy.schema._util import (  # noqa: E402
    receptor_json_is_membrane,
    receptor_json_is_presolvated,
)


def test_serialised_type_check_agrees_with_deserialising_one(tyk2_protein):
    """The cheap path must give the same answer as rebuilding the component."""
    payload = tyk2_protein.to_json()
    assert receptor_json_is_membrane(payload) is is_membrane_receptor(tyk2_protein)
    assert receptor_json_is_presolvated(payload) is is_presolvated_receptor(
        tyk2_protein
    )


@requires_membrane_openfe
def test_serialised_type_check_recognises_a_membrane_receptor(membrane_receptor):
    payload = membrane_receptor.to_json()
    assert receptor_json_is_membrane(payload) is True
    assert receptor_json_is_presolvated(payload) is True
    assert receptor_json_is_membrane(payload) is is_membrane_receptor(membrane_receptor)


@pytest.mark.parametrize("payload", ["not json", "", "[]", "null", "[[1,2,3]]"])
def test_serialised_type_check_returns_none_when_undeterminable(payload):
    """Unrecognised shapes must say 'I don't know' so callers fall back, not guess."""
    assert receptor_json_is_membrane(payload) is None
    assert receptor_json_is_presolvated(payload) is None


@requires_membrane_openfe
def test_is_membrane_does_not_deserialise_the_receptor(
    membrane_receptor, tyk2_ligands, monkeypatch
):
    """Regression: deserialising a real membrane receptor costs ~79 s; this must not."""
    from asapdiscovery.alchemy.schema.fec import (
        FreeEnergyCalculationFactory,
        FreeEnergyCalculationNetwork,
    )

    planned = FreeEnergyCalculationFactory().create_fec_dataset(
        dataset_name="cheap_is_membrane",
        receptor=membrane_receptor,
        ligands=tyk2_ligands[:3],
    )

    def _explode(self):
        raise AssertionError("is_membrane must not call to_openfe_receptor()")

    monkeypatch.setattr(FreeEnergyCalculationNetwork, "to_openfe_receptor", _explode)
    assert planned.is_membrane is True
