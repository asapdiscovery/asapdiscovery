"""Aligning a receptor to the membrane normal and locating the bilayer midplane.

Pure geometry: numpy in, numbers out, no OpenMM. That keeps the part of membrane building
most likely to be wrong -- and least likely to be noticed when it is -- independently
testable, without needing to build a 150,000-atom system to check it.

Coordinates are Angstrom throughout unless a name says ``_nm``.
"""

import numpy as np

from .schema import (
    BACKBONE_ATOMS,
    HYDROPHOBIC_RESIDUES,
    MembraneAlignmentSpec,
    ResidueRange,
)


def read_pdb_atoms(path) -> list[dict]:
    """Parse ATOM/HETATM records into dicts of chain, resi, resn, name, element, xyz.

    Deliberately not a full PDB parser: the build path needs exactly these fields, and
    reading them directly avoids depending on a toolkit's interpretation of a file that
    is about to be handed to OpenMM anyway.
    """
    atoms = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            atoms.append(
                {
                    "chain": line[21],
                    "resi": int(line[22:26]),
                    "resn": line[17:20].strip(),
                    "name": name,
                    # columns 76-78 are the element; fall back to the atom name's first
                    # character for files that leave them blank
                    "el": (line[76:78].strip() or name[:1]),
                    "xyz": np.array(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ]
                    ),
                }
            )
    return atoms


def rotation_to_z(axis, target=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation matrix taking unit ``axis`` onto ``target`` by the shortest arc (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    b = np.asarray(target, float)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-12:
        # already parallel (c>0) or antiparallel (c<0)
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def _ca_in_range(atoms, rng: ResidueRange) -> np.ndarray:
    return np.array(
        [
            a["xyz"]
            for a in atoms
            if a["name"] == "CA" and rng.contains(a["resi"], a["chain"])
        ]
    )


def membrane_axis(atoms, spec: MembraneAlignmentSpec):
    """The membrane-normal axis, plus an independent cross-check of it.

    The axis runs ``axis_from`` -> ``axis_to``, i.e. extracellular -> intracellular, from
    the CA centroids of two segments that lie on the axis by symmetry.

    The cross-check is the normal of the plane through the per-chain centroids. On a
    symmetric oligomer the two should agree closely; where they do not, it is usually
    because chains are unequally resolved, which biases the centroid method but not the
    landmark one. Both are reported so that disagreement is visible.

    Returns:
        (axis, crosscheck_normal, from_centroid, to_centroid)
    """
    start = _ca_in_range(atoms, spec.axis_from)
    end = _ca_in_range(atoms, spec.axis_to)
    if not len(start) or not len(end):
        raise ValueError(
            f"axis landmarks not found in structure: "
            f"axis_from residues {spec.axis_from.start}-{spec.axis_from.end} matched "
            f"{len(start)} CAs, axis_to {spec.axis_to.start}-{spec.axis_to.end} matched "
            f"{len(end)}. Check the residue numbering matches this structure."
        )

    axis = end.mean(0) - start.mean(0)
    axis /= np.linalg.norm(axis)

    cas = [a for a in atoms if a["name"] == "CA"]
    chains = sorted({a["chain"] for a in cas})
    centroids = np.array(
        [np.mean([a["xyz"] for a in cas if a["chain"] == c], axis=0) for c in chains]
    )
    if len(centroids) >= 3:
        normal = np.linalg.svd(centroids - centroids.mean(0))[2][2]
    else:
        normal = axis.copy()

    return axis, normal, start.mean(0), end.mean(0)


def membrane_placement(atoms, rot, origin, spec: MembraneAlignmentSpec) -> dict:
    """Locate the bilayer midplane along z, and measure the landmarks that verify it.

    The midplane is the mean midpoint of the z-spans of ``spec.midplane_helices``. It is
    deliberately *not* the protein centroid: an extracellular turret or a cytoplasmic
    domain drags a centroid well off the true midplane.

    Two standard heuristics are not used, because they are unreliable on exactly the
    proteins this is for. "Charged sidechains are excluded from the core, so find the
    depletion zone" fails on a voltage-gated channel, whose gating arginines sit inside
    the bilayer by design. "The two Trp/Tyr bands mark the headgroup interfaces" fails
    when a turret or pore helix contributes aromatics well above the membrane. What is
    used instead is the span of unambiguous lipid-facing helices, with the median depth
    of outward-facing hydrophobic sidechains kept as an independent cross-check.
    """

    def to_frame(a):
        return rot @ (a["xyz"] - origin)

    cas = [a for a in atoms if a["name"] == "CA"]

    spans = {}
    for label, rng in spec.tm_helices.items():
        z = np.array(
            [to_frame(a)[2] for a in cas if rng.contains(a["resi"], a["chain"])]
        )
        if len(z):
            spans[label] = {
                "z_min": round(float(z.min()), 2),
                "z_max": round(float(z.max()), 2),
                "midpoint": round(float((z.min() + z.max()) / 2), 2),
                "span": round(float(z.max() - z.min()), 2),
            }

    present = [h for h in spec.midplane_helices if h in spans]
    if not present:
        raise ValueError(
            f"none of the midplane helices {spec.midplane_helices} were found in the "
            f"structure (matched helices: {sorted(spans)}). Check residue numbering."
        )
    centre = float(np.mean([spans[h]["midpoint"] for h in present]))
    thickness = float(np.mean([spans[h]["span"] for h in present]))

    # cross-check: median z of outward-facing hydrophobic sidechain heavy atoms
    sidechain = [a for a in atoms if a["name"] not in BACKBONE_ATOMS and a["el"] != "H"]
    hyd_median = float("nan")
    if sidechain:
        xyz = np.array([to_frame(a) for a in sidechain])
        radial = np.linalg.norm(xyz[:, :2], axis=1)
        outer = radial >= np.percentile(radial, 65)
        is_hyd = (
            np.array([a["resn"] in HYDROPHOBIC_RESIDUES for a in sidechain]) & outer
        )
        if is_hyd.any():
            hyd_median = float(np.median(xyz[is_hyd, 2]))

    landmarks = {}
    for label, rng in spec.landmarks.items():
        z = np.array(
            [to_frame(a)[2] for a in cas if rng.contains(a["resi"], a["chain"])]
        )
        if len(z):
            landmarks[label] = round(float(z.mean()), 1)

    return {
        "membrane_centre_z": round(centre, 2),
        "method": f"midpoint of {'+'.join(present)} CA z-span (peripheral TM helices)",
        "hydrophobic_thickness_from_TM_span": round(thickness, 2),
        "crosscheck_outer_hydrophobic_median_z": round(hyd_median, 2),
        "crosscheck_offset": round(abs(hyd_median - centre), 2),
        "tm_helix_spans": spans,
        "landmark_z_after_centring": landmarks,
    }


def check_landmarks(placement: dict, spec: MembraneAlignmentSpec) -> list[str]:
    """Return a list of human-readable violations of ``spec.landmark_bounds``.

    Empty means the placement is consistent with everything the spec asserts about it.
    """
    violations = []
    measured = placement["landmark_z_after_centring"]
    for label, (lo, hi) in spec.landmark_bounds.items():
        if label not in measured:
            continue
        z = measured[label]
        if not (lo <= z <= hi):
            violations.append(
                f"{label} is at z={z:+.1f} A, outside the expected [{lo:+.1f}, {hi:+.1f}] A"
            )
    return violations


def best_inplane_angle(xyz_nm, patch_xy_nm, padding_nm):
    """Spin about z to minimise how many lipid patches the box needs.

    A rotation about z leaves the membrane normal exactly on z, so it is free as far as
    the barostat is concerned -- but not as far as cost is concerned. A tetrameric channel
    has a rounded-square footprint, so the axis-aligned bounding box shrinks markedly when
    the diagonal is put on the axes. Measured on hERG 8ZYP: 9.98 x 9.91 nm at 0 deg needs
    3x2 copies of the 6.37 x 6.69 nm POPC patch (255.6 nm^2), against 8.11 x 8.02 nm at
    40 deg which fits in 2x2 (170.4 nm^2). A third of the box -- and a third of the atom
    count, on every edge of every network -- for a free rotation.

    Returns:
        (key, degrees, (nx, ny), (extent_x_nm, extent_y_nm))
    """
    px, py = patch_xy_nm
    best = None
    for deg in np.arange(0, 90, 1.0):
        t = np.radians(deg)
        rot2d = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        projected = xyz_nm[:, :2] @ rot2d.T
        ex, ey = projected.max(0) - projected.min(0)
        nx = int(np.ceil((ex + 2 * padding_nm) / px))
        ny = int(np.ceil((ey + 2 * padding_nm) / py))
        key = (nx * px * ny * py, max(ex, ey))
        if best is None or key < best[0]:
            best = (key, float(deg), (nx, ny), (float(ex), float(ey)))
    return best


def structural_ion_positions(
    atoms, rot, origin, spec: MembraneAlignmentSpec
) -> list[dict]:
    """Positions (in the aligned frame) for ``spec.structural_ions``.

    Each ion goes at the midpoint of the backbone-carbonyl rings of its two cage residues.
    The coordination distances are returned alongside so the placement can be verified:
    a proper cage shows ~8 oxygens within 3.6 A.
    """
    out = []
    for site in spec.structural_ions:
        rings = {}
        for resi in (site.lower_residue, site.upper_residue):
            ring = np.array(
                [
                    rot @ (a["xyz"] - origin)
                    for a in atoms
                    if a["resi"] == resi and a["name"] == "O"
                ]
            )
            if len(ring):
                rings[resi] = ring
        if len(rings) != 2:
            continue
        lower, upper = rings[site.lower_residue], rings[site.upper_residue]
        pos = (lower.mean(0) + upper.mean(0)) / 2.0
        dists = np.sort(np.linalg.norm(np.vstack([lower, upper]) - pos, axis=1))
        out.append(
            {
                "site": site.name,
                "element": site.element,
                "xyz": pos,
                "n_coord_O": int((dists < 3.6).sum()),
                "coord_dist_A": [round(float(x), 2) for x in dists[:8]],
                "radial_offset_A": round(float(np.linalg.norm(pos[:2])), 2),
            }
        )
    return out
