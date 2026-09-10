"""Build a solvated, equilibrated lipid bilayer around a receptor.

Produces the single PDB that OpenFE's membrane workflow expects: a fully built, solvated,
pre-equilibrated system containing protein + lipids + water + ions and **no ligand**, with
explicit box vectors, aligned to the coordinate axes.

The alignment requirement is the one most easily missed. The complex leg runs under a
``MonteCarloMembraneBarostat``, which couples pressure anisotropically along z, so a pore
axis a few degrees off z means the barostat squeezes the bilayer at an angle -- a defect
that looks fine in a viewer.

Everything here goes through OpenMM's ``Modeller.addMembrane``; no PACKMOL-Memgen or
CHARMM-GUI is required.
"""

import json
import pathlib
import time
from collections import Counter
from typing import Optional

import numpy as np

from .geometry import (
    best_inplane_angle,
    check_landmarks,
    membrane_axis,
    membrane_placement,
    read_pdb_atoms,
    rotation_to_z,
    structural_ion_positions,
)
from .schema import MembraneAlignmentSpec, MembraneBuildSettings

#: Residue names that are never protein, used when separating protein from everything else.
_NON_PROTEIN = ("HOH", "WAT", "NA", "CL", "K", "LIG")


def lipid_patch(settings: MembraneBuildSettings):
    """OpenMM's pre-equilibrated lipid patch, with residues renamed for AMBER lipid17.

    The patch ships as ``openmm/app/data/<lipid>.pdb``, and PDB format allows only three
    characters for a residue name, so its lipids are called e.g. ``POP``. AMBER lipid17
    defines the template as ``POPC``, so ``addMembrane`` fails with "No template found for
    residue 1 (POP)". The atom names already agree (both use CHARMM-style naming), so
    renaming the residue is the whole fix.

    ``addMembrane`` accepts any object with ``.topology`` and ``.positions`` as
    ``lipidType``, which is the documented way to supply a custom patch.

    Returns:
        (patch, n_renamed)
    """
    from openmm import app

    path = pathlib.Path(app.__file__).parent / "data" / f"{settings.lipid}.pdb"
    if not path.exists():
        raise FileNotFoundError(
            f"OpenMM ships no pre-equilibrated patch for lipid {settings.lipid!r} "
            f"(looked for {path}). Available patches: "
            f"{sorted(p.stem for p in path.parent.glob('*.pdb'))}"
        )
    patch = app.PDBFile(str(path))
    short = settings.lipid[:3]
    n = 0
    if short != settings.lipid:
        for residue in patch.topology.residues():
            if residue.name == short:
                residue.name = settings.lipid
                n += 1
        if not n:
            raise RuntimeError(
                f"no {short!r} residues found in the {settings.lipid} patch to rename"
            )
    return patch, n


def composition(topology, settings: MembraneBuildSettings) -> dict:
    """Residue counts by category, for the build report."""
    counts = Counter(r.name for r in topology.residues())
    lipid_names = set(settings.lipid_resnames)
    protein = sum(
        v for k, v in counts.items() if k not in set(_NON_PROTEIN) | lipid_names
    )
    return {
        "protein_residues": protein,
        "lipids": sum(counts.get(n, 0) for n in lipid_names),
        "waters": counts.get("HOH", 0) + counts.get("WAT", 0),
        "Na": counts.get("NA", 0),
        "Cl": counts.get("CL", 0),
        "ligand": counts.get("LIG", 0),
        "atoms": topology.getNumAtoms(),
    }


def wrap_molecules(topology, positions, box, verbose: bool = True):
    """Translate each MOLECULE by whole box vectors until its centroid is in the box.

    Writing with ``enforcePeriodicBox=False`` keeps molecules intact but leaves every
    water where it diffused to, so after a couple of nanoseconds a large shell of bulk
    water sits outside the primary cell and is never brought back. Measured on a 2 ns
    system: 9.0% of atoms more than 0.75 box lengths from the origin, 1.0% beyond a full
    box length, the furthest at 1.75.

    That matters because ``MonteCarloMembraneBarostat`` rescales molecule centroids about
    the origin, so a molecule 1.7 box lengths out is displaced ~1.7x further per volume
    move than one at the centre -- the outer shell gets thrown around relative to its
    neighbours on every accepted move.

    Wrapping per *molecule* rather than per *atom* is what keeps this safe. Per-atom
    wrapping (``enforcePeriodicBox=True``) splits molecules across the boundary and,
    measured here, reported the leaflet split as 431 upper / 1 lower and the P-P thickness
    as 27.1 A against the true ~37 A. Centroids also leave the bilayer untouched: lipid
    centroids sit near z = +-1.5 nm in a ~8.7 nm box, nowhere near the +-L/2 boundary, so
    only bulk water and ions ever move.
    """
    from openmm import unit

    xyz = np.array(positions.value_in_unit(unit.nanometer))
    lengths = np.array([box[0, 0], box[1, 1], box[2, 2]], dtype=float)

    # molecules = connected components of the bond graph; isolated atoms (ions) are their own
    n_atoms = topology.getNumAtoms()
    parent = list(range(n_atoms))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for bond in topology.bonds():
        a, b = find(bond[0].index), find(bond[1].index)
        if a != b:
            parent[a] = b

    groups: dict[int, list[int]] = {}
    for i in range(n_atoms):
        groups.setdefault(find(i), []).append(i)

    moved = 0
    for indices in groups.values():
        idx = np.asarray(indices)
        shift = -np.round(xyz[idx].mean(axis=0) / lengths) * lengths
        if np.any(shift != 0.0):
            xyz[idx] += shift
            moved += len(idx)
    if verbose:
        print(
            f"    wrapped {moved} atoms in {len(groups)} molecules into the "
            "origin-centred box",
            flush=True,
        )
    return unit.Quantity(xyz, unit.nanometer)


def diagnostics(topology, positions, box, settings: MembraneBuildSettings) -> dict:
    """Area per lipid, bilayer thickness and leaflet balance.

    These are the numbers that say whether the bilayer is physically sane. Note that area
    per lipid is convention-dependent for a protein-containing bilayer -- the uncorrected
    and probe-corrected values bracket the experimental range rather than reproducing it
    -- so ``P_P_thickness_A`` is the unambiguous bilayer-quality number.
    """
    from openmm import unit

    xyz = np.array(positions.value_in_unit(unit.angstrom))
    atom_info = [(a.index, a.residue.name, a.name) for a in topology.atoms()]
    lipid_names = set(settings.lipid_resnames)

    # the phosphate P marks each headgroup; leaflets split by sign of z about the midpoint
    # of the P distribution, so the diagnostic is independent of any rigid translation
    phosphate_z = np.array(
        [xyz[i, 2] for i, rn, an in atom_info if rn in lipid_names and an == "P"]
    )
    mid = (
        float((phosphate_z.min() + phosphate_z.max()) / 2) if len(phosphate_z) else 0.0
    )
    upper, lower = phosphate_z[phosphate_z > mid], phosphate_z[phosphate_z < mid]
    n_up, n_lo = len(upper), len(lower)
    area_nm2 = float(box[0, 0] * box[1, 1])

    # protein cross-section through the hydrophobic core. Grid, not convex hull: splayed
    # voltage-sensor domains enclose inter-domain gaps that lipids genuinely occupy, and a
    # hull counts those as protein (measured 57.9 nm^2 -> 51.9 A^2 per lipid, below POPC's
    # experimental 64-68). A 1 A grid with a 4 A probe gives 48.7 nm^2 -> 56 A^2.
    protein_xy = np.array(
        [
            xyz[i, :2]
            for i, rn, an in atom_info
            if rn not in lipid_names | set(_NON_PROTEIN) and abs(xyz[i, 2] - mid) < 10.0
        ]
    )
    probe_a, cell_a = 4.0, 1.0
    protein_area_nm2 = 0.0
    if len(protein_xy) > 3:
        from scipy.spatial import cKDTree

        lo = protein_xy.min(0) - probe_a
        hi = protein_xy.max(0) + probe_a
        grid = np.stack(
            np.meshgrid(
                np.arange(lo[0], hi[0], cell_a),
                np.arange(lo[1], hi[1], cell_a),
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 2)
        occupied = (
            cKDTree(protein_xy).query_ball_point(grid, probe_a, return_length=True) > 0
        )
        protein_area_nm2 = float(occupied.sum() * cell_a * cell_a / 100.0)

    n_mean = max(1, (n_up + n_lo) / 2)
    return {
        "n_lipids_upper": n_up,
        "n_lipids_lower": n_lo,
        "leaflet_imbalance": abs(n_up - n_lo),
        "area_per_lipid_A2_uncorrected": (
            round(area_nm2 * 100 / n_mean, 1) if n_up + n_lo else None
        ),
        "protein_core_area_nm2": round(protein_area_nm2, 1),
        "protein_area_method": "1 A grid, 4 A probe, |z-mid|<10 A slab",
        "area_per_lipid_A2": (
            round((area_nm2 - protein_area_nm2) * 100 / n_mean, 1)
            if n_up + n_lo
            else None
        ),
        "P_P_thickness_A": (
            round(float(np.median(upper) - np.median(lower)), 1)
            if n_up and n_lo
            else None
        ),
        "leaflet_split_z_A": round(mid, 2),
        "box_xy_nm": [round(float(box[0, 0]), 3), round(float(box[1, 1]), 3)],
        "box_z_nm": round(float(box[2, 2]), 3),
    }


def align_receptor(
    receptor_pdb,
    spec: MembraneAlignmentSpec,
    settings: Optional[MembraneBuildSettings] = None,
):
    """Compute the transform putting the receptor in its membrane frame.

    The frame has the membrane normal on +z (extracellular up), the bilayer midplane at
    z=0, the protein centred in xy, and the footprint spun to its cheapest in-plane
    orientation.

    Returns:
        (atoms, rot, origin, info) -- ``rot @ (xyz - origin)`` maps into the membrane frame.

    Raises:
        ValueError: If the placement violates the spec's ``landmark_bounds``.
    """
    from openmm import unit as omm_unit

    settings = settings or MembraneBuildSettings()
    atoms = read_pdb_atoms(receptor_pdb)
    axis, normal, from_centroid, _ = membrane_axis(atoms, spec)

    # the axis runs extracellular->intracellular, so send it to -z to put extracellular at +z
    rot = rotation_to_z(-axis)

    # two passes: place the midplane from an uncentred frame, then re-measure in the final one
    first = membrane_placement(atoms, rot, from_centroid, spec)
    origin = from_centroid + np.linalg.inv(rot) @ np.array(
        [0, 0, first["membrane_centre_z"]]
    )

    # spin about z to the cheapest footprint, then centre the protein in xy
    patch, _ = lipid_patch(settings)
    box = np.array(
        patch.topology.getPeriodicBoxVectors().value_in_unit(omm_unit.nanometer)
    )
    heavy_nm = (
        np.array([rot @ (a["xyz"] - origin) for a in atoms if a["el"] != "H"]) / 10.0
    )
    _, degrees, n_patches, extent = best_inplane_angle(
        heavy_nm, (box[0, 0], box[1, 1]), settings.padding_nm
    )
    theta = np.radians(degrees)
    spin = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    rot = spin @ rot
    xy = np.array([rot @ (a["xyz"] - origin) for a in atoms if a["el"] != "H"])[:, :2]
    origin = origin + np.linalg.inv(rot) @ np.array([xy.mean(0)[0], xy.mean(0)[1], 0.0])

    placement = membrane_placement(atoms, rot, origin, spec)

    violations = check_landmarks(placement, spec)
    if violations:
        raise ValueError(
            "membrane placement failed the landmark checks defined by preset "
            f"{spec.name!r}:\n  "
            + "\n  ".join(violations)
            + "\nThis usually means the residue numbering in the structure does not match "
            "the spec, or the receptor is not the construct the spec describes."
        )

    z = np.array([(rot @ (a["xyz"] - origin))[2] for a in atoms if a["el"] != "H"])
    info = {
        "alignment_spec": spec.name,
        "n_atoms": len(atoms),
        "membrane_axis": [round(float(x), 4) for x in axis],
        "angle_axis_to_z_deg": round(
            float(np.degrees(np.arccos(abs(axis @ [0, 0, 1])))), 2
        ),
        "crosscheck_chain_normal": [round(float(x), 4) for x in normal],
        "axis_vs_normal_deg": round(
            float(np.degrees(np.arccos(np.clip(abs(normal @ axis), -1, 1)))), 2
        ),
        "belt": placement,
        "inplane": {
            "inplane_spin_deg": degrees,
            "patch_copies_xy": list(n_patches),
            "footprint_nm": [round(extent[0], 2), round(extent[1], 2)],
            "patch_xy_nm": [round(float(box[0, 0]), 3), round(float(box[1, 1]), 3)],
        },
        "protein_z_extent_after_align": [
            round(float(z.min()), 2),
            round(float(z.max()), 2),
        ],
    }
    return atoms, rot, origin, info


def build_membrane_system(
    receptor_pdb,
    spec: MembraneAlignmentSpec,
    output_prefix,
    equilibration_ns: float = 20.0,
    pocket_ligand=None,
    settings: Optional[MembraneBuildSettings] = None,
    verbose: bool = True,
) -> dict:
    """Build, solvate and equilibrate a bilayer around ``receptor_pdb``.

    Writes three files next to ``output_prefix``:

    * ``<prefix>_membrane_system.pdb`` -- the OpenFE input: no ligand, explicit box;
    * ``<prefix>_complex_equil.pdb``   -- the same system with the pocket ligand, for
      inspection;
    * ``<prefix>_membrane_report.json`` -- geometry, composition, equilibration log and
      bilayer diagnostics.

    About ``pocket_ligand``: OpenFE wants the deposited system *without* a ligand, but
    ``addMembrane`` also solvates, so waters land in the empty binding site and would then
    clash with whatever ligand is attached downstream. Passing the crystallographic ligand
    here keeps the pocket open during solvation and equilibration; it is stripped before
    the OpenFE input is written, leaving a ligand-shaped cavity rather than water. That is
    exactly right for the crystallographic pose and approximately right for a docked one.

    Args:
        receptor_pdb: Prepared receptor PDB (protein only, hydrogens added).
        spec: How to orient this receptor and where the midplane goes.
        output_prefix: Path prefix for the three output files.
        equilibration_ns: Total equilibration, including the restrained stages. Bilayers
            need far longer than a soluble protein: a 2 ns hERG bilayer was still visibly
            drifting (box z falling monotonically, PE down another ~10,000 kJ/mol over the
            last stage) and produced a ~30% task failure rate with SimulationNaNError,
            all of it in the complex leg. 20 ns is the validated default.
        pocket_ligand: Optional ligand file (SDF) to hold the binding site open.
        settings: Build settings; defaults reproduce the hERG production systems.
        verbose: Print progress.

    Returns:
        The build report, as also written to ``<prefix>_membrane_report.json``.
    """
    import openmm
    from openmm import app
    from openmm import unit as omm_unit

    settings = settings or MembraneBuildSettings()
    started = time.time()
    prefix = pathlib.Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    def say(message):
        if verbose:
            print(message, flush=True)

    atoms, rot, origin, info = align_receptor(receptor_pdb, spec, settings)
    belt = info["belt"]
    say(
        f"\n=== building {settings.lipid} bilayer around {pathlib.Path(receptor_pdb).name} ==="
    )
    say(
        f"  membrane axis was {info['angle_axis_to_z_deg']} deg off z, now aligned; "
        f"midplane from {belt['method']} (TM thickness "
        f"{belt['hydrophobic_thickness_from_TM_span']} A, cross-check offset "
        f"{belt['crosscheck_offset']} A)"
    )
    say(f"  landmarks after centring: {belt['landmark_z_after_centring']}")

    # ---- protein, rotated into the membrane frame
    pdb = app.PDBFile(str(receptor_pdb))
    positions = np.array(pdb.positions.value_in_unit(omm_unit.angstrom))
    positions = (rot @ (positions - origin).T).T
    modeller = app.Modeller(pdb.topology, positions * omm_unit.angstrom)

    forcefield = app.ForceField(*settings.forcefields)

    # ---- pocket-holding ligand, same transform
    ligand_name = None
    if pocket_ligand is not None:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        off_ligand = Molecule.from_file(str(pocket_ligand), allow_undefined_stereo=True)
        ligand_topology = off_ligand.to_topology().to_openmm()
        for residue in ligand_topology.residues():
            residue.name = "LIG"
        ligand_xyz = np.array(off_ligand.conformers[0].to("angstrom").magnitude)
        modeller.add(
            ligand_topology,
            ((rot @ (ligand_xyz - origin).T).T) * omm_unit.angstrom,
        )
        forcefield.registerTemplateGenerator(
            SMIRNOFFTemplateGenerator(
                molecules=[off_ligand], forcefield=settings.small_molecule_forcefield
            ).generator
        )
        ligand_name = pathlib.Path(pocket_ligand).stem
        say(
            f"  added {ligand_name} ({off_ligand.n_atoms} atoms) as a pocket placeholder"
        )

    # ---- structural ions the structure does not deposit
    ions = structural_ion_positions(atoms, rot, origin, spec)
    if ions:
        # built as its own Topology and merged with Modeller.add(), rather than appended to
        # modeller.positions -- appending turns the Quantity into a bare list, and
        # addMembrane then fails with "'list' object has no attribute 'value_in_unit'"
        ion_topology = app.Topology()
        chain = ion_topology.addChain()
        for ion in ions:
            element = app.element.get_by_symbol(ion["element"])
            residue = ion_topology.addResidue(ion["element"].upper(), chain)
            ion_topology.addAtom(ion["element"], element, residue)
        ion_positions = omm_unit.Quantity(
            [openmm.Vec3(*(ion["xyz"] / 10.0)) for ion in ions], omm_unit.nanometer
        )
        modeller.add(ion_topology, ion_positions)
        say(
            f"  added {len(ions)} structural ions at "
            + ", ".join(
                f"{i['site']}(z{i['xyz'][2]:+.1f}A, {i['n_coord_O']} O<3.6A)"
                for i in ions
            )
        )

    # ---- bilayer + water + ions. addMembrane tiles the pre-equilibrated patch, deletes
    # lipids clashing with the (temporarily shrunk) protein keeping the leaflets balanced,
    # then relaxes the membrane back around it.
    patch, n_renamed = lipid_patch(settings)
    say(
        f"  addMembrane(lipidType={settings.lipid} patch [{n_renamed} renamed], "
        f"membraneCenterZ=0, minimumPadding={settings.padding_nm} nm, "
        f"{settings.ion_concentration_molar} M NaCl) ..."
    )
    modeller.addMembrane(
        forcefield,
        lipidType=patch,
        membraneCenterZ=0 * omm_unit.nanometer,
        minimumPadding=settings.padding_nm * omm_unit.nanometer,
        ionicStrength=settings.ion_concentration_molar * omm_unit.molar,
        neutralize=True,
    )
    built = composition(modeller.topology, settings)
    say(f"  built: {built}")

    with open(f"{prefix}_complex_raw.pdb", "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)

    report = _equilibrate(
        modeller,
        forcefield,
        settings,
        equilibration_ns,
        prefix,
        say,
    )

    strip_topology, strip_positions, final_box = report.pop("_system")
    diag = diagnostics(strip_topology, strip_positions, final_box, settings)

    info.update(
        {
            "lipid": settings.lipid,
            "padding_nm": settings.padding_nm,
            "ion_molar": settings.ion_concentration_molar,
            "protein_forcefield": settings.forcefields,
            "small_molecule_forcefield": settings.small_molecule_forcefield,
            "pocket_placeholder": ligand_name,
            "composition_with_ligand": built,
            "composition_openfe_input": composition(strip_topology, settings),
            "box_vectors_nm": [[round(float(x), 5) for x in row] for row in final_box],
            "equilibration": report["equilibration"],
            "diagnostics": diag,
            "files": {
                "openfe_input": f"{prefix.name}_membrane_system.pdb",
                "inspection_complex": f"{prefix.name}_complex_equil.pdb",
                "pre_equilibration": f"{prefix.name}_complex_raw.pdb",
            },
            "coordinates_wrapped": True,
            "wall_time_s": round(time.time() - started, 1),
        }
    )
    pathlib.Path(f"{prefix}_membrane_report.json").write_text(
        json.dumps(info, indent=1)
    )
    say(f"  diagnostics: {diag}")
    say(
        f"  wrote {prefix}_membrane_system.pdb (+ complex + report) in "
        f"{info['wall_time_s']}s"
    )
    return info


def _equilibrate(modeller, forcefield, settings, equilibration_ns, prefix, say):
    """Minimise then equilibrate under a membrane barostat, tapering restraints.

    Restraints are released in stages so the lipids relax around a fixed protein before
    the protein itself is allowed to move.
    """
    import openmm
    from openmm import app
    from openmm import unit as omm_unit

    temperature = settings.temperature_kelvin * omm_unit.kelvin

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=settings.nonbonded_cutoff_nm * omm_unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=settings.hydrogen_mass * omm_unit.amu,
    )
    # semi-isotropic in xy, free in z, zero surface tension
    system.addForce(
        openmm.MonteCarloMembraneBarostat(
            settings.pressure_bar * omm_unit.bar,
            0.0 * omm_unit.bar * omm_unit.nanometer,
            temperature,
            openmm.MonteCarloMembraneBarostat.XYIsotropic,
            openmm.MonteCarloMembraneBarostat.ZFree,
            25,
        )
    )

    restraint = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter(
        "k", 10.0 * omm_unit.kilocalories_per_mole / omm_unit.angstrom**2
    )
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)

    lipid_names = set(settings.lipid_resnames)
    protein_residues = {
        r.index
        for r in modeller.topology.residues()
        if r.name not in lipid_names | set(_NON_PROTEIN) | {"LIG"}
    }
    n_restrained = 0
    for atom in modeller.topology.atoms():
        if (
            atom.residue.index in protein_residues
            and atom.element is not None
            and atom.element.symbol != "H"
        ):
            restraint.addParticle(
                atom.index,
                modeller.positions[atom.index].value_in_unit(omm_unit.nanometer),
            )
            n_restrained += 1
    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(
        temperature,
        1.0 / omm_unit.picosecond,
        settings.timestep_fs * omm_unit.femtosecond,
    )
    platform = (
        openmm.Platform.getPlatformByName(settings.platform)
        if settings.platform
        else None
    )
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    say(f"  minimising ({n_restrained} restrained protein heavy atoms) ...")
    before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(maxIterations=settings.minimization_steps)
    after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    say(
        f"    PE {before.value_in_unit(omm_unit.kilojoule_per_mole):.0f} -> "
        f"{after.value_in_unit(omm_unit.kilojoule_per_mole):.0f} kJ/mol"
    )

    simulation.context.setVelocitiesToTemperature(temperature)
    steps_per_ns = int(1e6 / settings.timestep_fs)

    restrained_ns = sum(stage.nanoseconds for stage in settings.restraint_schedule)
    schedule = [
        (stage.restraint_k, stage.nanoseconds) for stage in settings.restraint_schedule
    ]
    remainder = equilibration_ns - restrained_ns
    if remainder < 0:
        raise ValueError(
            f"equilibration_ns ({equilibration_ns}) is shorter than the restrained "
            f"schedule ({restrained_ns} ns). Increase it, or shorten "
            "settings.restraint_schedule."
        )
    schedule.append((0.0, remainder))

    log = []
    for force_constant, nanoseconds in schedule:
        if nanoseconds <= 0:
            continue
        simulation.context.setParameter(
            "k", force_constant * omm_unit.kilocalories_per_mole / omm_unit.angstrom**2
        )
        say(
            f"  equilibrating {nanoseconds:.2f} ns at restraint "
            f"k={force_constant} kcal/mol/A^2 ..."
        )
        simulation.step(max(1, int(nanoseconds * steps_per_ns)))
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        box = np.array(state.getPeriodicBoxVectors().value_in_unit(omm_unit.nanometer))
        log.append(
            {
                "restraint_k": force_constant,
                "ns": nanoseconds,
                "PE_kJ_per_mol": round(
                    float(
                        state.getPotentialEnergy().value_in_unit(
                            omm_unit.kilojoule_per_mole
                        )
                    ),
                    1,
                ),
                "box_nm": [round(float(box[i, i]), 4) for i in range(3)],
            }
        )
        say(f"    PE {log[-1]['PE_kJ_per_mol']:.0f} kJ/mol, box {log[-1]['box_nm']} nm")

    # enforcePeriodicBox=True would wrap every ATOM into [0,L), splitting molecules across
    # the boundary and moving the bilayer off z=0. Keep molecules whole and wrap by
    # molecule instead -- see wrap_molecules().
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
    final_box = np.array(
        state.getPeriodicBoxVectors().value_in_unit(omm_unit.nanometer)
    )
    topology = simulation.topology
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    final_positions = wrap_molecules(
        topology, state.getPositions(), final_box, verbose=bool(say)
    )

    # inspection copy: everything, ligand included
    with open(f"{prefix}_complex_equil.pdb", "w") as fh:
        app.PDBFile.writeFile(topology, final_positions, fh, keepIds=True)

    # OpenFE input: ligand stripped, leaving a ligand-shaped cavity rather than water
    stripped = app.Modeller(topology, final_positions)
    stripped.delete([r for r in stripped.topology.residues() if r.name == "LIG"])
    stripped.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with open(f"{prefix}_membrane_system.pdb", "w") as fh:
        app.PDBFile.writeFile(stripped.topology, stripped.positions, fh, keepIds=True)

    return {
        "equilibration": {
            "ns_total": equilibration_ns,
            "timestep_fs": settings.timestep_fs,
            "barostat": "MonteCarloMembraneBarostat XYIsotropic/ZFree",
            "stages": log,
        },
        "_system": (stripped.topology, stripped.positions, final_box),
    }
