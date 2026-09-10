"""Schemas describing how to embed a receptor in a lipid bilayer.

Membrane embedding cannot be inferred reliably from a structure alone, so it is described
declaratively here and then *validated against* the built system rather than assumed.

A :class:`MembraneAlignmentSpec` says three things about a receptor:

* which residues define the membrane normal (``axis_from`` / ``axis_to``);
* which helices bracket the hydrophobic core (``midplane_helices``), and so where the
  bilayer midplane sits;
* which residues must land at a known depth if the placement is right (``landmarks``).

The third is what turns "the membrane looks about right" into a check that can fail.
See :data:`HERG` for a worked example.
"""

from typing import Literal, Optional

from pydantic import Field, model_validator

from ..schema.base import _SchemaBase

#: Residue names treated as hydrophobic when cross-checking the midplane against the
#: median depth of outward-facing sidechains.
HYDROPHOBIC_RESIDUES = frozenset({"LEU", "ILE", "VAL", "PHE", "MET", "ALA"})

#: Backbone atom names, excluded from that sidechain cross-check.
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


class ResidueRange(_SchemaBase):
    """An inclusive residue-number range, optionally restricted to one chain."""

    type: Literal["ResidueRange"] = "ResidueRange"

    start: int = Field(..., description="First residue number, inclusive.")
    end: int = Field(..., description="Last residue number, inclusive.")
    chain: Optional[str] = Field(
        None,
        description="Restrict to this chain id. None means every chain, which is what "
        "you want for a symmetric oligomer: the range is then averaged over all copies.",
    )

    @model_validator(mode="after")
    def _check_order(self):
        if self.end < self.start:
            raise ValueError(
                f"residue range end ({self.end}) is before start ({self.start})"
            )
        return self

    def contains(self, resi: int, chain: Optional[str] = None) -> bool:
        """True if residue ``resi`` (on ``chain``) falls in this range."""
        if self.chain is not None and chain is not None and chain != self.chain:
            return False
        return self.start <= resi <= self.end


class StructuralIonSite(_SchemaBase):
    """A constructed ion position, defined by the cage of backbone carbonyls around it.

    Some channels coordinate ions at sites that are structurally essential but frequently
    absent from deposited coordinates. Such an ion is placed at the midpoint of two
    consecutive backbone-carbonyl rings, and its coordination distances are reported so
    the placement can be checked rather than asserted.
    """

    type: Literal["StructuralIonSite"] = "StructuralIonSite"

    name: str = Field(..., description="Label for the site, e.g. 'S2'.")
    element: str = Field("K", description="Element symbol of the ion.")
    lower_residue: int = Field(
        ..., description="Residue whose backbone O forms the inner ring of the cage."
    )
    upper_residue: int = Field(
        ..., description="Residue whose backbone O forms the outer ring of the cage."
    )


class MembraneAlignmentSpec(_SchemaBase):
    """How to orient a receptor in a bilayer and where to put the midplane.

    Alignment is not cosmetic. The complex leg runs under a ``MonteCarloMembraneBarostat``,
    which couples pressure anisotropically along z, so a pore axis a few degrees off z
    means the barostat squeezes the bilayer at an angle.
    """

    type: Literal["MembraneAlignmentSpec"] = "MembraneAlignmentSpec"

    name: str = Field(..., description="Name of this spec, e.g. 'herg'.")
    description: str = Field("", description="What this spec describes.")

    axis_from: ResidueRange = Field(
        ...,
        description="Residues whose CA centroid marks the EXTRACELLULAR end of the "
        "membrane-normal axis. Pick a segment lying on the axis by symmetry that is "
        "resolved in every chain.",
    )
    axis_to: ResidueRange = Field(
        ...,
        description="Residues whose CA centroid marks the INTRACELLULAR end of the axis. "
        "The axis runs from `axis_from` to `axis_to` and is mapped onto -z, so that "
        "extracellular ends up at +z.",
    )
    tm_helices: dict[str, ResidueRange] = Field(
        ...,
        description="Transmembrane helices, by label. Only those named in "
        "`midplane_helices` set the midplane; the rest are measured and reported.",
    )
    midplane_helices: list[str] = Field(
        ...,
        description="Which `tm_helices` bracket the hydrophobic core. Use unambiguous "
        "single-pass, lipid-facing helices resolved in every chain. Avoid helices that "
        "are not purely lipid-facing -- a voltage sensor, for instance, carries charged "
        "residues through the core by design.",
    )
    landmarks: dict[str, ResidueRange] = Field(
        default_factory=dict,
        description="Residues whose depth is diagnostic once the midplane is set, e.g. a "
        "known drug-binding residue that must come out mid-bilayer. Reported in the "
        "build report, and checked against `landmark_bounds` where those are given.",
    )
    landmark_bounds: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Optional (min_z, max_z) in Angstrom for a landmark, measured from "
        "the midplane. A landmark outside its bounds fails the build rather than "
        "producing a quietly misplaced bilayer.",
    )
    structural_ions: list[StructuralIonSite] = Field(
        default_factory=list,
        description="Ion sites to construct, for channels whose filter ions are not "
        "deposited. Adjacent cages are only ~3.3 A apart, so list an alternating subset.",
    )

    @model_validator(mode="after")
    def _check_helices(self):
        missing = [h for h in self.midplane_helices if h not in self.tm_helices]
        if missing:
            raise ValueError(
                f"midplane_helices {missing} are not defined in tm_helices "
                f"{sorted(self.tm_helices)}"
            )
        if not self.midplane_helices:
            raise ValueError("at least one midplane helix is required")
        unknown = [k for k in self.landmark_bounds if k not in self.landmarks]
        if unknown:
            raise ValueError(
                f"landmark_bounds reference undefined landmarks {unknown}; defined "
                f"landmarks are {sorted(self.landmarks)}"
            )
        return self


def _rr(start: int, end: int) -> ResidueRange:
    return ResidueRange(start=start, end=end)


#: hERG (KCNH2) transmembrane module, as used for the OpenADMET hERG benchmark.
#:
#: The axis is the selectivity filter (S624-V625-G626-F627-G628, the SVGFG motif) to the
#: intracellular end of S6. Both segments are resolved in all four chains of 8ZYO and
#: 8ZYP, which whole-chain centroids are not: 8ZYO chain C is missing all of S2-S3.
#:
#: The midplane comes from S1 and S2 only. S4 is excluded deliberately -- it is the
#: voltage sensor, and its gating arginines sit inside the bilayer by design, so the
#: usual "charged residues are excluded from the core" heuristic finds no depletion zone
#: at all on this protein.
#:
#: The landmark bounds encode what a correct hERG placement looks like: Tyr652 and
#: Phe656, the classic drug-binding residues on S6, must come out mid-bilayer, and the
#: selectivity filter just inside the outer interface.
HERG = MembraneAlignmentSpec(
    name="herg",
    description=(
        "hERG (KCNH2) transmembrane module, residues ~407-665, tetramer chains A-D. "
        "Pore axis from the selectivity filter to the intracellular S6 ends."
    ),
    axis_from=_rr(624, 628),
    axis_to=_rr(658, 665),
    tm_helices={
        "S1": _rr(407, 426),
        "S2": _rr(452, 471),
        "S4": _rr(520, 545),
        "S6": _rr(640, 665),
    },
    midplane_helices=["S1", "S2"],
    landmarks={
        "turret_582_597": _rr(582, 597),
        "selectivity_filter_624_628": _rr(624, 628),
        "Tyr652": _rr(652, 652),
        "Phe656": _rr(656, 656),
        "S6_end_658_665": _rr(658, 665),
    },
    landmark_bounds={
        # measured 8ZYP / 8ZYO after centring: +30.7 / +30.1
        "turret_582_597": (20.0, 45.0),
        # measured: +15.3 / +14.6
        "selectivity_filter_624_628": (8.0, 22.0),
        # measured: +1.0 / +0.2 -- mid-bilayer, the drug site
        "Tyr652": (-8.0, 8.0),
        # measured: -2.7 / -3.8
        "Phe656": (-10.0, 6.0),
        # measured: -9.8 / -11.1
        "S6_end_658_665": (-20.0, -2.0),
    },
    structural_ions=[
        # S2 and S4 are the alternating pair actually coordinated in these structures;
        # S1 is not a cage at all (nearest O 4.3 A) because hERG's outer filter is
        # genuinely wide. Nothing is placed in the central cavity -- that is the ligand
        # site. Adjacent cages are ~3.3 A apart, so filling both would be absurd.
        StructuralIonSite(name="S4", element="K", lower_residue=624, upper_residue=625),
        StructuralIonSite(name="S2", element="K", lower_residue=626, upper_residue=627),
    ],
)


#: Named alignment specs shipped with ASAP-Alchemy, by lowercase name.
PRESETS: dict[str, MembraneAlignmentSpec] = {HERG.name: HERG}


def get_preset(name: str) -> MembraneAlignmentSpec:
    """Return the shipped :class:`MembraneAlignmentSpec` called ``name``.

    Raises:
        KeyError: If no such preset exists.
    """
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown membrane preset {name!r}; available presets are {sorted(PRESETS)}. "
            "Supply a MembraneAlignmentSpec JSON file instead to describe a different "
            "receptor."
        )


class EquilibrationStage(_SchemaBase):
    """One stage of restraint-tapering equilibration."""

    type: Literal["EquilibrationStage"] = "EquilibrationStage"

    restraint_k: float = Field(
        ...,
        ge=0.0,
        description="Positional restraint force constant on protein heavy atoms, in "
        "kcal/mol/A^2. 0 means unrestrained.",
    )
    nanoseconds: float = Field(..., gt=0.0, description="Length of this stage in ns.")


class MembraneBuildSettings(_SchemaBase):
    """Everything about building the bilayer that is not receptor geometry.

    Defaults reproduce the OpenADMET hERG production systems exactly.
    """

    type: Literal["MembraneBuildSettings"] = "MembraneBuildSettings"

    lipid: str = Field(
        "POPC",
        description="Lipid type. Must be one OpenMM ships a pre-equilibrated patch for "
        "(openmm/app/data/<lipid>.pdb) and that the protein force field parameterises.",
    )
    padding_nm: float = Field(
        1.5,
        gt=0,
        description="Minimum padding between the protein and the box edge, in nm. Match "
        "this to the FEC factory's solvent_padding.",
    )
    ion_concentration_molar: float = Field(
        0.15, ge=0, description="NaCl concentration in molar."
    )
    forcefields: list[str] = Field(
        default_factory=lambda: [
            "amber14/protein.ff14SB.xml",
            "amber14/lipid17.xml",
            "amber14/tip3p.xml",
        ],
        description="OpenMM force field XMLs for the build. NOTE the amber14/ prefix on "
        "lipid17: 'amber/lipid17.xml' also loads without error but supplies NO lipid "
        "template, and addMembrane then dies with a misleading 'No template found for "
        "residue 1 (POPC) ... the bonds are different'.",
    )
    small_molecule_forcefield: str = Field(
        "openff-2.2.0.offxml",
        description="Force field for the pocket-holding ligand, if one is used. Match "
        "the FEC factory's small_molecule_forcefield.",
    )
    temperature_kelvin: float = Field(
        298.15, gt=0, description="Equilibration temperature."
    )
    pressure_bar: float = Field(1.0, gt=0, description="Equilibration pressure.")
    timestep_fs: float = Field(
        4.0, gt=0, description="Integration timestep in fs (with hydrogen mass 3 amu)."
    )
    hydrogen_mass: float = Field(
        3.0, gt=0, description="Hydrogen mass in amu, for the 4 fs timestep."
    )
    nonbonded_cutoff_nm: float = Field(
        0.9, gt=0, description="PME real-space cutoff in nm."
    )
    minimization_steps: int = Field(
        5000, gt=0, description="Max minimisation iterations."
    )
    restraint_schedule: list[EquilibrationStage] = Field(
        default_factory=lambda: [
            EquilibrationStage(restraint_k=10.0, nanoseconds=0.2),
            EquilibrationStage(restraint_k=2.5, nanoseconds=0.2),
            EquilibrationStage(restraint_k=0.5, nanoseconds=0.2),
        ],
        description="Restrained stages run before the unrestrained remainder. Lipids "
        "relax around a fixed protein first, then the protein is released gradually.",
    )
    surface_tension_bar_nm: float = Field(
        0.0,
        description="Surface tension for the MonteCarloMembraneBarostat, in bar*nm. Zero "
        "is the usual choice for an equilibrium bilayer, and matches what the FEC "
        "protocol applies to the complex leg.",
    )
    random_seed: Optional[int] = Field(
        None,
        description="Seed for the initial velocity assignment. Set it to make a build "
        "reproducible; None lets OpenMM choose, so repeated builds differ.",
    )
    platform: Optional[str] = Field(
        "CUDA",
        description="OpenMM platform name. None picks the fastest available -- useful for "
        "testing, but a real build wants a GPU.",
    )

    @property
    def lipid_resnames(self) -> tuple[str, ...]:
        """Both spellings of the lipid residue name.

        PDB format truncates residue names to three characters, so a system read back
        from disk carries e.g. ``POP`` while a live OpenMM topology carries ``POPC``.
        Anything selecting lipids must accept both or it silently matches zero -- which
        is exactly how one hERG build reported leaflets 0/0 and a protein core area of
        205 nm^2, the 432 unmatched lipids having fallen through into the protein
        selection.
        """
        short = self.lipid[:3]
        return (self.lipid,) if short == self.lipid else (self.lipid, short)
