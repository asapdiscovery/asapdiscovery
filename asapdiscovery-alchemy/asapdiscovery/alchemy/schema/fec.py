import json
import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, TypeAlias

import gufe
import openfe
from alchemiscale import ScopedKey
from gufe import settings
from gufe.settings.models import SettingsBaseModel
from gufe.settings.typing import (
    GufeQuantity,
    KCalPerMolQuantity,
    NanometerQuantity,
    specify_quantity_units,
)
from gufe.tokenization import JSON_HANDLER, GufeKey
from openfe.setup.atom_mapping import lomap_scorers, perses_scorers
from openff.units import unit as OFFUnit
from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from ._util import check_ligand_series_uniqueness_and_names
from .base import _SchemaBase, _SchemaBaseFrozen
from .network import NetworkPlanner, PlannedNetwork
from .protocols import (
    available_protocols,
    build_protocol,
    default_protocol_settings,
    is_node_protocol,
    needs_solvent_leg,
)

if TYPE_CHECKING:
    from cinnabar import FEMap
    from gufe.mapping import LigandAtomMapping

    from asapdiscovery.data.schema.ligand import Ligand

MolarQuantity: TypeAlias = Annotated[GufeQuantity, specify_quantity_units("molar")]

# the flat OpenFE-RFE settings fields used by the pre-multi-protocol ("legacy")
# FreeEnergyCalculationFactory/Network format; these fold into a single
# RelativeHybridTopologyProtocolSettings under the current schema
_LEGACY_RFE_SETTING_FIELDS = (
    "forcefield_settings",
    "thermo_settings",
    "solvation_settings",
    "alchemical_settings",
    "engine_settings",
    "integrator_settings",
    "simulation_settings",
    "lambda_settings",
    "protocol_repeats",
    "partial_charge_settings",
    "output_settings",
)


class SolventSettings(_SchemaBase):
    """
    A settings class to encode the solvent used in the OpenFE FEC calculations.
    """

    type: Literal["SolventSettings"] = "SolventSettings"

    smiles: str = Field("O", description="The smiles pattern of the solvent.")
    positive_ion: str = Field(
        "Na+",
        description="The positive monoatomic ion which should be used to neutralize the system and to adjust the ionic concentration.",
    )
    negative_ion: str = Field(
        "Cl-",
        description="The negative monoatomic ion which should be used to neutralize the system and to adjust the ionic concentration.",
    )
    neutralize: bool = Field(
        True,
        description="If the net charge of the chemical system should be neutralized by the ions defined by `positive_ion` and `negative_ion`.",
    )
    ion_concentration: MolarQuantity = Field(
        0.15 * OFFUnit.molar,
        description="The ionic concentration required in molar units.",
    )

    def to_solvent_component(self) -> gufe.SolventComponent:
        return gufe.SolventComponent(**{k: v for k, v in self if k != "type"})


class AdaptiveSettings(_SchemaBase):
    """
    A settings class to encode settings for adaptive settings. These were recommended by OpenFE.
    """

    type: Literal["AdaptiveSettings"] = "AdaptiveSettings"
    adaptive_sampling: bool = Field(
        False,
        description="If True, will enable increase in production length of simulations given a `adaptive_sampling_multiplier` and `adaptive_sampling_threshold`.",
    )
    adaptive_sampling_multiplier: float = Field(
        2.0,
        description="The number of times more production simulation length (sampling time) that will be assigned to edges whose mapping scoring falls below the `adaptive_sampling_threshold`.",
    )
    adaptive_sampling_threshold: float = Field(
        0.5,
        description="The threshold that separates edges that are expected to perform well (higher; regular production simulation time) and poorly (lower; regular production simulation time * `adaptive_sampling_multiplier`). Recommended settings are 0.5 (LOMAP scorer) or 0.85 (PERSES scorer).",
    )
    adaptive_solvent_padding: bool = Field(
        True,
        description="Whether or not to use adaptive solvent padding; typically the complex phase can handle smaller padding size.",
    )
    solvent_padding_complex: NanometerQuantity = Field(
        1.5 * OFFUnit.nanometer,
        description="The solvent padding (in nm) to use for the complex phase of each edge.",
    )
    solvent_padding_solvated: NanometerQuantity = Field(
        1.5 * OFFUnit.nanometer,
        description="The solvent padding (in nm) to use for the solvated phase of each edge.",
    )

    def get_adapted_sampling_settings(
        self,
        scorer_method: str,
        mapping: "LigandAtomMapping",
        settings: "gufe.settings.Settings",
        base_sampling_length: OFFUnit.Quantity,
        sim_settings_attr: str = "simulation_settings",
    ) -> "gufe.settings.Settings":
        """
        It's advisable to increase simulation time on edges that are expected to be less reliable. There
        Aren't many good estimators for this, but the network planner edge scoring is a decent approximation.

        If the edge scoring (computed using `scorer_method`) is below the `adaptive_sampling_threshold` the
        simulation time is multiplied by `adaptive_sampling_multiplier`. Just to be sure, we use the base
        protocol's sampling time and not the provided edge protocol sampling time as a base value.

        ``sim_settings_attr`` names the attribute on ``settings`` that holds the
        simulation-time sub-settings (e.g. ``"simulation_settings"`` for RFE, or
        ``"complex_simulation_settings"`` / ``"solvent_simulation_settings"`` for
        protocols with per-leg settings such as ``SepTopProtocol`` or
        ``AbsoluteBindingProtocol``).

        Mutates and returns the (editable) protocol settings.
        """
        if scorer_method == "default_lomap":
            scorer = lomap_scorers.default_lomap_score
        elif scorer_method == "default_perses":
            scorer = perses_scorers.default_perses_scorer
        else:
            raise ValueError(
                f"Atom mapping scorer {scorer_method} not recognized; use one of `default_lomap`, `default_perses`."
            )
        if scorer(mapping) < self.adaptive_sampling_threshold:
            getattr(settings, sim_settings_attr).production_length = (
                base_sampling_length * self.adaptive_sampling_multiplier
            )
        return settings

    def get_adapted_solvent_settings(
        self,
        leg: str,
        settings: "gufe.settings.Settings",
        solv_settings_attr: str = "solvation_settings",
    ) -> "gufe.settings.Settings":
        """
        Certain water box shapes (such as dodecahedron) are able to handle slightly smaller padding size
        in the complex phase compared to the solvated phase. Given the leg (either "solvent" or "complex")
        this method applies the specified padding per phase (`solvent_padding_solvated` or
        `solvent_padding_complex`, resp.).

        ``solv_settings_attr`` names the solvation sub-settings attribute to mutate
        (e.g. ``"solvation_settings"`` for RFE, or ``"complex_solvation_settings"`` /
        ``"solvent_solvation_settings"`` for protocols with per-leg settings such as
        ``SepTopProtocol`` or ``AbsoluteBindingProtocol``).

        Mutates and returns the (editable) protocol settings.
        """
        padding = (
            self.solvent_padding_solvated
            if leg == "solvent"
            else self.solvent_padding_complex
        )
        getattr(settings, solv_settings_attr).solvent_padding = padding
        return settings

    def apply_settings(
        self,
        edge_settings: "gufe.settings.Settings",
        network_scorer: str,
        mapping: "LigandAtomMapping",
        leg: str,
        base_settings: "gufe.settings.Settings",
    ) -> "gufe.settings.Settings":
        """
        Applies a set of adaptive settings to a protocol's settings if requested.

        Operates on (and returns) the protocol ``Settings`` directly so the caller
        can build a fresh ``Protocol`` from them, rather than deep-copying a
        ``Protocol`` object. ``edge_settings`` is expected to be an editable
        (``unfrozen_copy``) settings object which is mutated in place.

        Adaptive settings are only applied where the settings expose the relevant
        fields. ``RelativeHybridTopologyProtocol`` settings carry a flat
        ``simulation_settings`` (sampling length) and ``solvation_settings``
        (solvent padding). ``SepTopProtocol`` and ``AbsoluteBindingProtocol``
        use per-leg split attributes (``complex_simulation_settings`` /
        ``solvent_simulation_settings`` and the equivalent solvation attributes).
        Non-equilibrium protocols (e.g. feflow's ``NonEquilibriumCyclingProtocol``)
        have no ``simulation_settings`` so adaptive sampling is skipped with a warning.
        """
        # double the simulation time if requested
        if self.adaptive_sampling:
            if hasattr(edge_settings, "simulation_settings") and hasattr(
                base_settings, "simulation_settings"
            ):
                # flat settings: RelativeHybridTopologyProtocol style
                base_sampling_length = (
                    base_settings.simulation_settings.production_length
                )
                edge_settings = self.get_adapted_sampling_settings(
                    network_scorer, mapping, edge_settings, base_sampling_length
                )
            else:
                # split per-leg settings: SepTopProtocol / AbsoluteBindingProtocol style
                leg_sim_attr = f"{leg}_simulation_settings"
                if hasattr(edge_settings, leg_sim_attr) and hasattr(
                    base_settings, leg_sim_attr
                ):
                    base_sampling_length = getattr(
                        base_settings, leg_sim_attr
                    ).production_length
                    edge_settings = self.get_adapted_sampling_settings(
                        network_scorer,
                        mapping,
                        edge_settings,
                        base_sampling_length,
                        sim_settings_attr=leg_sim_attr,
                    )
                else:
                    warnings.warn(
                        "adaptive_sampling requested but protocol settings "
                        f"{type(edge_settings).__name__} have no `simulation_settings` "
                        f"or `{leg}_simulation_settings`; "
                        "skipping adaptive sampling for this protocol."
                    )

        # adjust solvent padding per phase if requested
        if self.adaptive_solvent_padding:
            if hasattr(edge_settings, "solvation_settings"):
                # flat settings: RelativeHybridTopologyProtocol style
                edge_settings = self.get_adapted_solvent_settings(leg, edge_settings)
            else:
                # split per-leg settings: SepTopProtocol / AbsoluteBindingProtocol style
                leg_solv_attr = f"{leg}_solvation_settings"
                if hasattr(edge_settings, leg_solv_attr):
                    edge_settings = self.get_adapted_solvent_settings(
                        leg, edge_settings, solv_settings_attr=leg_solv_attr
                    )
                else:
                    warnings.warn(
                        "adaptive_solvent_padding requested but protocol settings "
                        f"{type(edge_settings).__name__} have no `solvation_settings` "
                        f"or `{leg}_solvation_settings`; "
                        "skipping adaptive solvent padding for this protocol."
                    )

        return edge_settings


# TODO make base class with abstract methods to collect results.
class TransformationResult(_SchemaBaseFrozen):
    """
    Store the results of a transformation, note when retries are used this will be the average result.
    """

    type: Literal["TransformationResult"] = "TransformationResult"
    ligand_a: str = Field(
        ..., description="The name of the ligand in state A of the transformation."
    )
    ligand_b: Optional[str] = Field(
        None,
        description="The name of the ligand in state B of the transformation. "
        "``None`` for node-based (ABFE) results where state B contains no ligand.",
    )
    phase: Literal["complex", "solvent", "combined"] = Field(
        ...,
        description=(
            "The phase of the transformation.  ``'combined'`` is used for protocols "
            "that handle both complex and solvent legs internally and return ΔΔG "
            "directly (e.g. SepTopProtocol)."
        ),
    )
    estimate: KCalPerMolQuantity = Field(
        ..., description="The average estimate of this transformation in kcal/mol"
    )
    uncertainty: KCalPerMolQuantity = Field(
        ...,
        description="The standard deviation of the estimates of this transform in kcal/mol",
    )
    protocol: Optional[str] = Field(
        None,
        description="The name of the alchemical protocol that produced this result. "
        "May be None for results collected before multi-protocol support.",
    )

    def name(self):
        """Make a name for this transformation based on the names of the ligands."""
        if self.ligand_b is None:
            return self.ligand_a
        return "-".join([self.ligand_a, self.ligand_b])


class _BaseResults(_SchemaBaseFrozen):
    """
    A base results class which handles the collecting and processing of the results.
    """

    type: Literal["_BaseResults"] = "_BaseResults"
    results: list[TransformationResult] = Field(
        [], description="The list of results collected for this dataset."
    )

    @property
    def protocols(self) -> list[Optional[str]]:
        """The distinct protocols present in the collected results.

        Order follows first appearance in ``results``. Legacy results without a
        protocol tag appear as ``None``.
        """
        seen = []
        for result in self.results:
            if result.protocol not in seen:
                seen.append(result.protocol)
        return seen

    def to_cinnabar_measurements(self, protocol: Optional[str] = "__all__"):
        """
        Combine the solvent and complex phases of each transformation into cinnabar
        measurement objects.

        For edge-based (RBFE) results (``ligand_b`` is set) a
        ``cinnabar.Measurement`` (relative ΔΔG) is produced from
        ``ΔG_complex − ΔG_solvent``.  For node-based (ABFE) results
        (``ligand_b is None``) a ``cinnabar.AbsoluteMeasurement`` (absolute ΔG)
        is produced from the same combination.

        Args:
            protocol: If provided (including ``None`` for legacy results), only
                results produced by that protocol are used. The default sentinel
                ``"__all__"`` uses every result; results are always grouped by
                ``(protocol, transformation name)`` so different protocols for the
                same edge are never merged.

        Returns:
            A list of ``cinnabar.Measurement`` and/or ``cinnabar.AbsoluteMeasurement``
            objects made from the combined solvent and complex phases.
        """
        from collections import defaultdict

        import numpy as np
        from cinnabar import Measurement

        if protocol == "__all__":
            results = self.results
        else:
            results = [r for r in self.results if r.protocol == protocol]

        raw_results = defaultdict(list)
        # gather by (protocol, transform) so distinct protocols are kept separate
        for result in results:
            raw_results[(result.protocol, result.name())].append(result)

        # Validate phase completeness and separate "combined" (SepTop) edges from
        # two-leg (RFE/NEQ/ABFE) edges.
        keys_to_remove = []
        for key, transforms in raw_results.items():
            _, name = key
            phases = {t.phase for t in transforms}
            if "combined" in phases:
                # Single-Transformation protocols (e.g. SepTopProtocol) return ΔΔG
                # directly; expect exactly one result per edge.
                if len(transforms) != 1:
                    raise RuntimeError(
                        f"The transformation {name} has {len(transforms)} results with "
                        f"phase='combined'; expected exactly one."
                    )
            else:
                missing_phase = {"complex", "solvent"} - phases
                if missing_phase:
                    warnings.warn(
                        f"The transformation {name} is missing simulated legs in the following phases {missing_phase}; removing"
                    )
                    keys_to_remove.append(key)
                elif len(transforms) > 2:
                    raise RuntimeError(
                        f"The transformation {name} has too many simulated legs, found the following phases {[t.phase for t in transforms]} expected complex and solvent."
                    )

        for key in keys_to_remove:
            raw_results.pop(key)

        # make the cinnabar data
        all_results = []
        for transforms in raw_results.values():
            phases = {t.phase for t in transforms}
            if "combined" in phases:
                # Single-Transformation protocols return the final free energy directly
                # from get_estimate() — no leg subtraction required.
                # SepTop → ΔΔG_bind (RBFE); AbsoluteBinding → ΔG_bind (ABFE).
                combined = transforms[0]
                if combined.ligand_b is None:
                    # ABFE: absolute ΔG_bind
                    try:
                        from cinnabar import AbsoluteMeasurement

                        result = AbsoluteMeasurement(
                            label=combined.ligand_a,
                            DG=combined.estimate,
                            uncertainty=combined.uncertainty,
                            computational=True,
                            source="calculated",
                        )
                    except ImportError:
                        result = Measurement(
                            labelA=combined.ligand_a,
                            labelB="__vacuum__",
                            DG=combined.estimate,
                            uncertainty=combined.uncertainty,
                            computational=True,
                            source="calculated",
                        )
                else:
                    # SepTop (and similar): relative ΔΔG_bind
                    result = Measurement(
                        labelA=combined.ligand_a,
                        labelB=combined.ligand_b,
                        DG=combined.estimate,
                        uncertainty=combined.uncertainty,
                        computational=True,
                        source="calculated",
                    )
            else:
                leg1, leg2 = transforms
                complex_leg: TransformationResult = (
                    leg1 if leg1.phase == "complex" else leg2
                )
                solvent_leg: TransformationResult = (
                    leg1 if leg1.phase == "solvent" else leg2
                )
                dg = complex_leg.estimate - solvent_leg.estimate
                uncertainty = np.sqrt(
                    complex_leg.uncertainty**2 + solvent_leg.uncertainty**2
                )

                if leg1.ligand_b is None:
                    # ABFE: absolute ΔG_bind — use cinnabar AbsoluteMeasurement so it
                    # can serve as an anchor in a combined RBFE+ABFE FEMap
                    try:
                        from cinnabar import AbsoluteMeasurement

                        result = AbsoluteMeasurement(
                            label=leg1.ligand_a,
                            DG=dg,
                            uncertainty=uncertainty,
                            computational=True,
                            source="calculated",
                        )
                    except ImportError:
                        # fall back to a relative measurement with a sentinel second label
                        # if the installed cinnabar version predates AbsoluteMeasurement
                        result = Measurement(
                            labelA=leg1.ligand_a,
                            labelB="__vacuum__",
                            DG=dg,
                            uncertainty=uncertainty,
                            computational=True,
                            source="calculated",
                        )
                else:
                    # RBFE: relative ΔΔG_bind
                    result = Measurement(
                        labelA=leg1.ligand_a,
                        labelB=leg1.ligand_b,
                        DG=dg,
                        uncertainty=uncertainty,
                        computational=True,
                        source="calculated",
                    )
            all_results.append(result)
        return all_results

    def to_fe_map(self, protocol: Optional[str] = "__all__"):
        """
        Convert the set of relative free energy estimates to a cinnabar FEMap object to calculate the absolute values
        or plot vs experiment.

        Args:
            protocol: If provided (including ``None``), restrict the map to results
                from that protocol. The default uses every result.

        Returns:
            A cinnabar.FEMap made from the relative results objects.
        """
        from cinnabar import FEMap

        fe_graph = FEMap()
        for result in self.to_cinnabar_measurements(protocol=protocol):
            fe_graph.add_measurement(measurement=result)
        return fe_graph

    def to_fe_map_by_protocol(self) -> dict[Optional[str], "FEMap"]:
        """Build one cinnabar FEMap per protocol present in the results.

        Returns:
            A mapping of protocol name (or ``None`` for legacy results) to the
            FEMap built from only that protocol's results.
        """
        return {
            protocol: self.to_fe_map(protocol=protocol) for protocol in self.protocols
        }


class AlchemiscaleResults(_BaseResults):
    type: Literal["AlchemiscaleResults"] = "AlchemiscaleResults"

    network_key: ScopedKey = Field(
        ...,
        description="The alchemiscale key associated with this submited network, which is used to gather results from the client.",
    )


class _FreeEnergyBase(_SchemaBase):
    """
    A base class for the FreeEnergyCalculationFactory and Network to work around the serialisation issues with
    openFE settings models see <https://github.com/OpenFreeEnergy/openfe/issues/518>.
    """

    type: Literal["_FreeEnergyBase"] = "_FreeEnergyBase"

    solvent_settings: SolventSettings = Field(
        SolventSettings(),
        description="The solvent settings which should be used during the free energy calculations.",
    )
    adaptive_settings: Optional[AdaptiveSettings] = Field(
        AdaptiveSettings(),
        description="Run adaptive settings depending on e.g. expected edge reliability or system phase.",
    )
    protocol: list[str] = Field(
        default_factory=lambda: ["RelativeHybridTopologyProtocol"],
        description="The list of alchemical protocols to use. Each must be a protocol "
        "registered in `asapdiscovery.alchemy.schema.protocols`, e.g. "
        "'RelativeHybridTopologyProtocol', 'NonEquilibriumCyclingProtocol', or "
        "'FahNonEquilibriumCyclingProtocol'.",
    )
    protocol_strategy: Literal["all"] = Field(
        "all",
        description="The strategy used to assign protocols to ligand transformations. "
        "'all' creates a Transformation between every ligand mapping for each "
        "included protocol.",
    )
    protocol_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of protocol name to its `ProtocolSettings` object. "
        "Auto-populated with the default settings of each protocol listed in "
        "`protocol` when not provided.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_flat_format(cls, data: Any) -> Any:
        """Raise a clear error when loading a pre-multi-protocol (flat) file.

        Older factory/network files stored the OpenFE-RFE settings as flat
        top-level fields and ``protocol`` as a plain string. Those fields no
        longer exist, so without this guard loading such a file would fail with
        an opaque ``extra="forbid"`` pydantic error on real user data.
        """
        if isinstance(data, dict):
            found = sorted(set(_LEGACY_RFE_SETTING_FIELDS).intersection(data))
            if found:
                raise ValueError(
                    "This file was created with a pre-multi-protocol version of "
                    f"asapdiscovery-alchemy (found legacy settings fields: {found}). "
                    "These files are no longer supported. Convert it with "
                    "`asap-alchemy convert-network`, regenerate it with "
                    "`asap-alchemy create` / `asap-alchemy prep create`, or rebuild "
                    "the network from its original inputs."
                )
        return data

    @field_validator("protocol_settings", mode="before")
    @classmethod
    def _decode_protocol_settings(cls, value: Any) -> dict[str, Any]:
        """Decode serialized protocol settings back into gufe ``Settings`` objects.

        Values may already be live ``Settings`` objects (in-memory construction) or
        JSON-safe dicts produced by :meth:`_encode_protocol_settings` on load; the
        latter carry gufe class markers that ``JSON_HANDLER`` uses to reconstruct
        the concrete ``Settings`` subclass.
        """
        if not isinstance(value, dict):
            return value
        decoded = {}
        for name, settings_obj in value.items():
            if isinstance(settings_obj, SettingsBaseModel):
                decoded[name] = settings_obj
            else:
                decoded[name] = json.loads(
                    json.dumps(settings_obj), cls=JSON_HANDLER.decoder
                )
        return decoded

    @field_serializer("protocol_settings", mode="plain")
    def _encode_protocol_settings(self, value: dict[str, Any]) -> dict[str, Any]:
        """Encode gufe ``Settings`` objects into self-describing JSON-safe dicts.

        Pydantic would otherwise flatten the ``Settings`` into plain dicts and lose
        the gufe class markers needed to reconstruct the correct subclass. We
        pre-encode with ``JSON_HANDLER`` so the markers survive ``to_file``.

        Note: ``_SchemaBase.to_file`` re-runs ``JSON_HANDLER`` over the whole
        ``model_dump`` output, but by then these values are already marker-bearing
        plain dicts with no gufe objects left, so that second pass is a no-op here.
        """
        return {
            name: json.loads(json.dumps(settings_obj, cls=JSON_HANDLER.encoder))
            for name, settings_obj in value.items()
        }

    @model_validator(mode="after")
    def _populate_default_protocol_settings(self) -> "_FreeEnergyBase":
        """Validate protocol names and auto-populate any missing default settings."""
        known = available_protocols()
        for name in self.protocol:
            if name not in known:
                raise ValueError(
                    f"Unknown protocol {name!r}; available protocols are {known}."
                )
            # mutate the dict's contents in place rather than reassigning the
            # field, so this also works on the frozen `FreeEnergyCalculationNetwork`
            # subclass (where attribute assignment is disallowed)
            if name not in self.protocol_settings:
                self.protocol_settings[name] = default_protocol_settings(name)
        return self

    @property
    def small_molecule_forcefield(self) -> str:
        """The small molecule force field shared by the configured protocols.

        Pulled from the ``forcefield_settings`` of each protocol that exposes them;
        raises if the protocols disagree so callers (bespoke fitting, bespoke
        parameter injection) get a single, unambiguous force field.

        Note: this makes bespoke fitting unavailable for genuinely mixed-force-field
        multi-protocol networks; per-edge injection in ``to_alchemical_network`` uses
        each protocol's own force field directly and does not rely on this helper.
        """
        ff_names = {
            self.protocol_settings[name].forcefield_settings.small_molecule_forcefield
            for name in self.protocol
            if hasattr(self.protocol_settings[name], "forcefield_settings")
        }
        if not ff_names:
            raise ValueError(
                "None of the configured protocols expose a small molecule force field."
            )
        if len(ff_names) > 1:
            raise ValueError(
                "The configured protocols use inconsistent small molecule force "
                f"fields: {sorted(ff_names)}."
            )
        return ff_names.pop()

    def to_openfe_protocols(self) -> dict[str, "gufe.Protocol"]:
        """Build a mapping of protocol name to its instantiated ``gufe.Protocol``."""
        return {
            name: build_protocol(name, self.protocol_settings[name])
            for name in self.protocol
        }


class FreeEnergyCalculationNetwork(_FreeEnergyBase):
    """
    A schema of a FEC network created by the FreeEnergyCalculationFactory which contains all runtime settings and can
    be converted to local openFE inputs or submitted to alchemiscale.
    """

    type: Literal["FreeEnergyCalculationNetwork"] = "FreeEnergyCalculationNetwork"
    dataset_name: str = Field(
        ...,
        description="The name of the dataset, this will be used for local files and the alchemiscale network.",
    )
    network: PlannedNetwork = Field(
        ...,
        description="The planned free energy network with atom mappings between ligands.",
    )
    receptor: str = Field(
        ...,
        description="The JSON str of the receptor which should be used in the FEC calculation.",
    )
    results: Optional[AlchemiscaleResults] = Field(
        None,
        description="The results object which tracks how the calculation was run locally or on alchemiscale and stores the physical results.",
    )
    experimental_protocol: Optional[str] = Field(
        None,
        description="The name of the experimental protocol in the CDD vault that should be associated with this Alchemy network.",
    )
    target: Optional[str] = Field(
        None,
        description="The name of the biological target associated with this Alchemy network.",
    )

    model_config = ConfigDict(frozen=True, from_attributes=True)

    def to_openfe_receptor(self) -> openfe.ProteinComponent:
        return openfe.ProteinComponent.from_json(content=self.receptor)

    def _protocols_for_edge(self, mapping: "LigandAtomMapping") -> list[str]:
        """Select which protocols to apply to a given ligand mapping (edge).

        Driven by ``protocol_strategy``:

        - ``"all"``: every configured *edge-based* protocol is applied to every
          edge.  Node-based protocols (ABFE) are excluded — they are applied per
          ligand node in :meth:`_protocols_for_node`.

        Future strategies (e.g. assigning a single protocol per edge based on edge
        properties) can use ``mapping`` to make that decision here.
        """
        if self.protocol_strategy == "all":
            return [p for p in self.protocol if not is_node_protocol(p)]
        raise NotImplementedError(
            f"protocol_strategy {self.protocol_strategy!r} is not implemented."
        )

    def _protocols_for_node(self) -> list[str]:
        """Return the node-based (ABFE) protocols to apply to each ligand node.

        Node-based protocols generate one ``Transformation`` per ligand rather
        than per ligand pair, so they are handled separately from edge protocols.
        """
        return [p for p in self.protocol if is_node_protocol(p)]

    def to_alchemical_network(self) -> openfe.AlchemicalNetwork:
        """
        Create an openfe AlchemicalNetwork from the planned network which can be submitted to alchemiscale or ran locally

        Returns:
            An openfe.AlchemicalNetwork created from the schema.
        """
        transformations = []
        # do all openfe conversions
        ligand_network = self.network.to_ligand_network()
        solvent = self.solvent_settings.to_solvent_component()
        receptor = self.to_openfe_receptor()

        # build the network; `protocol_strategy` decides which protocols apply to
        # each edge (the "all" strategy uses every configured protocol per edge)
        for mapping in ligand_network.edges:
            for protocol_name in self._protocols_for_edge(mapping):
                base_settings = self.protocol_settings[protocol_name]
                # compute the (possibly bespoke) force field once per edge/protocol,
                # using this protocol's own base force field; leg-independent
                ff_string = None
                if hasattr(base_settings, "forcefield_settings"):
                    ff_string = self._inject_bespoke_parameters(
                        edge=mapping,
                        base_force_field=base_settings.forcefield_settings.small_molecule_forcefield,
                    )

                # Protocols that handle both legs internally (e.g. SepTopProtocol)
                # require a single complex-phase Transformation only; their
                # get_estimate() already returns ΔΔG = ΔG_complex − ΔG_solvent.
                legs = ["complex"] if not needs_solvent_leg(protocol_name) else ["solvent", "complex"]
                for leg in legs:
                    sys_a_dict = {"ligand": mapping.componentA, "solvent": solvent}
                    sys_b_dict = {"ligand": mapping.componentB, "solvent": solvent}
                    if leg == "complex":
                        sys_a_dict["protein"] = receptor
                        sys_b_dict["protein"] = receptor

                    system_a = openfe.ChemicalSystem(
                        sys_a_dict, name=f"{mapping.componentA.name}_{leg}"
                    )
                    system_b = openfe.ChemicalSystem(
                        sys_b_dict, name=f"{mapping.componentB.name}_{leg}"
                    )

                    # build a fresh, editable copy of this protocol's settings for
                    # this edge/leg rather than deep-copying a gufe Protocol object
                    edge_settings = base_settings.unfrozen_copy()
                    if ff_string is not None:
                        edge_settings.forcefield_settings.small_molecule_forcefield = (
                            ff_string
                        )

                    # run this edge's settings through adaptive settings. If this list of things to pass
                    # grows any larger we should only pass the `FreeEnergyCalculationNetwork` and instead
                    # infer these parameters somewhere in self.adaptive_settings.
                    if self.adaptive_settings:
                        edge_settings = self.adaptive_settings.apply_settings(
                            edge_settings,  # the editable settings to be adjusted
                            self.network.scorer,  # the network edge scorer - for adaptive sampling
                            mapping,  # the atom mapping for this edge - for adaptive sampling
                            leg,  # whether this edge is complex or solvated phase - for adaptive solvent box padding
                            base_settings,  # base settings to compare with for internal checking
                        )

                    # build a fresh Protocol from the per-edge settings instead of
                    # mutating a deep-copied one (keeps gufe's flyweight intact)
                    edge_protocol = build_protocol(protocol_name, edge_settings)

                    # set up the transformation; the protocol name is appended so
                    # transformations for the same edge but different protocols have
                    # distinct, human-readable names (and so results can be mapped
                    # back to their protocol)
                    transformation = openfe.Transformation(
                        stateA=system_a,
                        stateB=system_b,
                        mapping={"ligand": mapping},
                        protocol=edge_protocol,  # use protocol created above
                        name=f"{system_a.name}_{system_b.name}_{protocol_name}",
                    )
                    transformations.append(transformation)

        # node-based (ABFE) protocols: one Transformation per ligand.
        # AbsoluteBindingProtocol (and similar) runs both complex and solvent legs
        # internally from a single Transformation and returns ΔG_bind directly from
        # get_estimate().  Its _validate_endstates() requires ProteinComponent in both
        # stateA and stateB, so stateB is the apo protein (no ligand), not a bare
        # solvent box.
        for protocol_name in self._protocols_for_node():
            base_settings = self.protocol_settings[protocol_name]

            for ligand in ligand_network.nodes:
                sys_a = openfe.ChemicalSystem(
                    {"ligand": ligand, "protein": receptor, "solvent": solvent},
                    name=f"{ligand.name}_bound",
                )
                # stateB: ligand is annihilated; protein + solvent remain.
                sys_b = openfe.ChemicalSystem(
                    {"protein": receptor, "solvent": solvent},
                    name=f"{ligand.name}_apo",
                )
                transformations.append(
                    openfe.Transformation(
                        stateA=sys_a,
                        stateB=sys_b,
                        mapping=None,
                        protocol=build_protocol(protocol_name, base_settings),
                        name=f"{ligand.name}_{protocol_name}",
                    )
                )

        return openfe.AlchemicalNetwork(edges=transformations, name=self.dataset_name)

    def _inject_bespoke_parameters(
        self, edge: "LigandAtomMapping", base_force_field: str
    ) -> str:
        """
        Inject the bespoke torsion parameters for the given edge into the base force field.

        Args:
            edge: The edge from the OpenFE alchemical network which we want the parameters for.
            base_force_field: The small molecule force field of the protocol this edge
                is being built for, into which any bespoke parameters are injected.

        Returns:
            The string of the force field with bespoke parameters added or the name of the base force field if no
            bespoke parameters are found

        Notes:
            They will always be added in the order of the mapping (ligandA, ligandB)
        """
        from openff.toolkit import ForceField
        from openff.toolkit.utils.exceptions import DuplicateParameterError
        from openff.units import unit

        # get the name of the base ff and load it
        ff_string = base_force_field
        if ".offxml" not in ff_string:
            ff_string += ".offxml"
        ff = ForceField(ff_string)

        # map the names to ligands to quickly find the parameters
        names_to_ligands = {
            ligand.compound_name: ligand for ligand in self.network.ligands
        }

        # torsion data to manually set the phase idivf and periodicity
        torsion_data = {
            "idivf1": 1.0,
            "idivf2": 1.0,
            "idivf3": 1.0,
            "idivf4": 1.0,
            "phase1": 0.0 * unit.degree,
            "phase2": 180 * unit.degree,
            "phase3": 0 * unit.degree,
            "phase4": 180 * unit.degree,
            "periodicity1": 1,
            "periodicity2": 2,
            "periodicity3": 3,
            "periodicity4": 4,
        }

        # track if we have any bespoke parameters
        bespoke_parameters = False
        for ofe_ligand in [edge.componentA, edge.componentB]:
            if (
                edge_ligand := names_to_ligands[ofe_ligand.name]
            ).bespoke_parameters is not None:
                bespoke_parameters = True
                for parameter in edge_ligand.bespoke_parameters.parameters:
                    handler = ff.get_parameter_handler(parameter.interaction)
                    parameter_data = {
                        key: value * getattr(unit, parameter.units)
                        for key, value in parameter.values.items()
                    }
                    parameter_data["smirks"] = parameter.smirks
                    parameter_data["id"] = f"bespokefit_{edge_ligand.compound_name}"
                    if parameter.interaction == "ProperTorsions":
                        parameter_data.update(torsion_data)
                    try:
                        # similar ligands will share parameters so make sure we don't add it twice
                        handler.add_parameter(parameter_kwargs=parameter_data)
                    except DuplicateParameterError:
                        continue

        # if we found bespoke parameters return the new force field
        if bespoke_parameters:
            ff_string = ff.to_string()

        return ff_string


def convert_legacy_fec_network(data: dict) -> "FreeEnergyCalculationNetwork":
    """Convert a legacy (pre-multi-protocol) network dict to the current schema.

    Old-style ``FreeEnergyCalculationNetwork`` files stored the OpenFE-RFE settings
    as flat top-level fields with ``protocol`` as a plain string and untagged
    results. This folds those flat settings into a single
    ``RelativeHybridTopologyProtocolSettings`` under
    ``protocol_settings["RelativeHybridTopologyProtocol"]``, sets ``protocol`` to a
    one-element list, and tags each result with that protocol.

    Args:
        data: The ``gufe.tokenization.JSON_HANDLER``-decoded contents of a legacy
            network file (i.e. ``json.load(f, cls=JSON_HANDLER.decoder)``).

    Returns:
        An equivalent ``FreeEnergyCalculationNetwork`` in the current schema.

    Raises:
        ValueError: If ``data`` does not look like a legacy flat-format network.
    """
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocolSettings

    if not isinstance(data, dict):
        raise ValueError("Expected a decoded network dict to convert.")

    missing = [field for field in _LEGACY_RFE_SETTING_FIELDS if field not in data]
    if missing:
        raise ValueError(
            "Input does not look like a legacy (pre-multi-protocol) "
            f"FreeEnergyCalculationNetwork; missing expected flat settings fields: "
            f"{missing}."
        )

    protocol_name = "RelativeHybridTopologyProtocol"
    rfe_settings = RelativeHybridTopologyProtocolSettings(
        **{field: data[field] for field in _LEGACY_RFE_SETTING_FIELDS}
    )

    # carry results forward, tagging each with the single legacy protocol
    results = data.get("results")
    if results is not None:
        for result in results.get("results", []):
            result.setdefault("protocol", protocol_name)

    # only forward optional/defaulted fields that are actually present so missing
    # ones fall back to the current schema defaults
    optional = {
        field: data[field]
        for field in (
            "solvent_settings",
            "adaptive_settings",
            "experimental_protocol",
            "target",
        )
        if field in data
    }

    return FreeEnergyCalculationNetwork(
        dataset_name=data["dataset_name"],
        network=data["network"],
        receptor=data["receptor"],
        protocol=[protocol_name],
        protocol_strategy="all",
        protocol_settings={protocol_name: rfe_settings},
        results=results,
        **optional,
    )


class FreeEnergyCalculationFactory(_FreeEnergyBase):
    """A factory class to configure FEC calculations using the openFE pipeline. This generates a prepared FEC network
    which can be executed locally or submitted to Alchemiscale."""

    type: Literal["FreeEnergyCalculationFactory"] = "FreeEnergyCalculationFactory"

    network_planner: NetworkPlanner = Field(
        NetworkPlanner(),
        description="The network planner settings which should be used to construct the network.",
    )

    def create_fec_dataset(
        self,
        dataset_name: str,
        receptor: openfe.ProteinComponent,
        ligands: Optional[list["Ligand"]] = None,
        central_ligand: Optional["Ligand"] = None,
        graphml: Optional[str] = None,
        experimental_protocol: Optional[str] = None,
        target: Optional[str] = None,
    ) -> FreeEnergyCalculationNetwork:
        """
         Use the factory settings to create a FEC dataset using OpenFE models.

        Args:
            dataset_name: The name which should be given to this dataset, this will be used for local file creation or
                to identify on alchemiscale
            receptor: The prepared receptor to use in the FEC dataset.
            ligands: The list of prepared and state enumerated ligands to use in the FEC calculation.
            central_ligand: An optional ligand which should be considered as the center only needed for radial networks.
                Note this ligand will be deduplicated from the list if it appears in both.
            experimental_protocol: The name of the experimental protocol in the CDD vault that should be
                associated with this Alchemy network.
            target: The name of the biological target associated with this Alchemy network.

         Returns:
             The planned FEC network which can be executed locally or submitted to alchemiscale.
        """
        # generate the network
        if ligands:
            check_ligand_series_uniqueness_and_names(ligands)
            # start by trying to plan the network
            planned_network = self.network_planner.generate_network(
                ligands=ligands,
                central_ligand=central_ligand,
            )
        # pre-generated network
        elif graphml:
            # equivalent name checks in constructor
            planned_network = PlannedNetwork.from_graphml(graphml)

        else:
            raise ValueError("Either ligands or a graphml file must be provided.")

        planned_fec_network = FreeEnergyCalculationNetwork(
            dataset_name=dataset_name,
            network=planned_network,
            receptor=receptor.to_json(),
            experimental_protocol=experimental_protocol,
            target=target,
            **self.model_dump(exclude={"type", "network_planner"}),
        )
        return planned_fec_network


class _BaseFailure(_SchemaBaseFrozen):
    """Base class for collecting errors and tracebacks from failed FEC runs"""

    type: Literal["_BaseFailure"] = "_BaseFailure"

    error: tuple[str, tuple[Any, ...]] = Field(
        tuple(), description="Exception raised and associated message."
    )
    traceback: str = Field(
        "", description="Complete traceback associated with the failure."
    )


class AlchemiscaleFailure(_BaseFailure):
    """Class for collecting errors and tracebacks from errored tasks in an alchemiscale network"""

    type: Literal["AlchemiscaleFailure"] = "AlchemiscaleFailure"

    network_key: ScopedKey = Field(
        ...,
        description="The alchemiscale key associated with this submitted network, which is used to gather the failed results from the client.",
    )
    task_key: ScopedKey = Field(..., description="Task key for the errored task.")
    unit_key: GufeKey = Field(
        ..., description="Protocol unit key associated to the errored task."
    )
    dag_result_key: GufeKey = Field(
        ..., description="Protocol DAG result key associated to the errored task."
    )

    @field_validator("unit_key", "dag_result_key", mode="before")
    @classmethod
    def _coerce_gufe_key(cls, value):
        """Coerce plain strings into ``GufeKey``.

        ``GufeKey`` is a ``str`` subclass, so under pydantic v2 the field is
        validated with an ``is_instance_of`` check. Keys embedded directly in
        a ``GufeTokenizable`` (e.g. ``ProtocolUnitFailure.source_key``) lose
        their type on round-trip and come back from alchemiscale as plain
        ``str``, which would otherwise fail that check.

        Workaround for upstream gufe bug:
        https://github.com/OpenFreeEnergy/gufe/issues/713
        Once that is fixed, this validator can be removed.
        """
        if isinstance(value, str) and not isinstance(value, GufeKey):
            return GufeKey(value)
        return value
