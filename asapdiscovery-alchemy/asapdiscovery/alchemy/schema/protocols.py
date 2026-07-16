"""Registry of alchemical ``Protocol``\\ s available to ASAP-Alchemy.

This module provides a small index that maps a protocol *name* to the module and
class that implement it, together with helpers to instantiate the protocol, fetch
its default settings, and reverse-map a protocol object back to its registered
name.

The protocols live in optional dependencies (``feflow`` and ``alchemiscale-fah``
are not hard requirements of ``asapdiscovery-alchemy``), so all imports are
performed lazily and a helpful :class:`ImportError` is raised if a requested
protocol's package is not installed.
"""

import importlib
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import gufe
    from gufe.settings import Settings


class _ProtocolRegistration(NamedTuple):
    """How to locate a protocol implementation and the package that provides it."""

    module: str
    protocol_class: str
    #: The distribution that must be installed to use this protocol (for error messages).
    package: str
    #: True for node-based protocols (ABFE: one Transformation per ligand, no mapping);
    #: False for edge-based protocols (RBFE: one Transformation per ligand *pair*).
    is_node_protocol: bool = False
    #: True when the protocol requires separate complex and solvent Transformations
    #: (e.g. RFE, NEQ); False when the protocol handles both legs internally from a
    #: single complex-phase Transformation and returns ΔΔG directly (e.g. SepTop).
    needs_solvent_leg: bool = True


#: Index of the alchemical protocols ASAP-Alchemy knows how to build.
#: Keyed by the protocol *class name*, which is also the public identifier used in
#: the ``protocol`` field of the FEC factory/network schema.
PROTOCOL_REGISTRY: dict[str, _ProtocolRegistration] = {
    "RelativeHybridTopologyProtocol": _ProtocolRegistration(
        module="openfe.protocols.openmm_rfe",
        protocol_class="RelativeHybridTopologyProtocol",
        package="openfe",
    ),
    "SepTopProtocol": _ProtocolRegistration(
        module="openfe.protocols.openmm_septop",
        protocol_class="SepTopProtocol",
        package="openfe",
        needs_solvent_leg=False,
    ),
    "NonEquilibriumCyclingProtocol": _ProtocolRegistration(
        module="feflow.protocols",
        protocol_class="NonEquilibriumCyclingProtocol",
        package="feflow",
    ),
    "FahNonEquilibriumCyclingProtocol": _ProtocolRegistration(
        module="alchemiscale_fah.protocols.feflow",
        protocol_class="FahNonEquilibriumCyclingProtocol",
        package="alchemiscale-fah",
    ),
    "AbsoluteBindingProtocol": _ProtocolRegistration(
        module="openfe.protocols.openmm_afe",
        protocol_class="AbsoluteBindingProtocol",
        package="openfe",
        is_node_protocol=True,
        needs_solvent_leg=False,
    ),
}


def needs_solvent_leg(name: str) -> bool:
    """Return ``True`` if ``name`` requires a separate solvent-phase Transformation.

    Most RBFE protocols (RFE, NEQ) need two ``Transformation``s per edge — one
    complex and one solvent — and combine them as ΔΔG = ΔG_complex − ΔG_solvent.
    ``SepTopProtocol`` runs both legs inside a single complex-phase ``Transformation``
    and returns ΔΔG directly from ``get_estimate()``, so no separate solvent
    ``Transformation`` should be created.

    Raises:
        KeyError: If ``name`` is not a registered protocol.
    """
    try:
        return PROTOCOL_REGISTRY[name].needs_solvent_leg
    except KeyError:
        raise KeyError(
            f"Unknown protocol {name!r}; available protocols are "
            f"{available_protocols()}."
        )


def is_node_protocol(name: str) -> bool:
    """Return ``True`` if ``name`` is a node-based (ABFE) protocol.

    Node-based protocols generate one ``Transformation`` per *ligand* (no
    partner ligand and no atom mapping). Edge-based protocols generate one
    ``Transformation`` per *ligand pair*.

    Raises:
        KeyError: If ``name`` is not a registered protocol.
    """
    try:
        return PROTOCOL_REGISTRY[name].is_node_protocol
    except KeyError:
        raise KeyError(
            f"Unknown protocol {name!r}; available protocols are "
            f"{available_protocols()}."
        )


def available_protocols() -> list[str]:
    """Return the names of all protocols registered with ASAP-Alchemy.

    Note this lists the protocols ASAP-Alchemy *knows about*; a given protocol is
    only usable if its providing package is installed (see
    :func:`get_protocol_class`).
    """
    return list(PROTOCOL_REGISTRY)


def get_protocol_class(name: str) -> "type[gufe.Protocol]":
    """Return the ``gufe.Protocol`` subclass registered under ``name``.

    Args:
        name: The registered protocol name, e.g. ``"RelativeHybridTopologyProtocol"``.

    Raises:
        KeyError: If ``name`` is not a registered protocol.
        ImportError: If the package providing the protocol is not installed.
    """
    try:
        registration = PROTOCOL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown protocol {name!r}; available protocols are "
            f"{available_protocols()}."
        )
    try:
        module = importlib.import_module(registration.module)
    except ImportError as error:
        raise ImportError(
            f"The protocol {name!r} requires the {registration.package!r} package, "
            f"which is not installed (could not import {registration.module!r}). "
            f"Install it to use this protocol."
        ) from error
    return getattr(module, registration.protocol_class)


def _asap_relative_hybrid_topology_settings() -> "Settings":
    """ASAP-tuned default settings for ``RelativeHybridTopologyProtocol``.

    These reproduce the defaults ASAP-Alchemy historically applied on top of
    OpenFE's defaults (force field, dodecahedral box, gapsys softcore, ASAP
    simulation lengths, and a single protocol repeat) so that the multi-protocol
    refactor does not silently change production behavior for the RFE protocol.
    """
    from openff.units import unit as OFFUnit

    protocol_class = get_protocol_class("RelativeHybridTopologyProtocol")
    protocol_settings = protocol_class.default_settings()
    protocol_settings = protocol_settings.unfrozen_copy()

    protocol_settings.forcefield_settings.small_molecule_forcefield = (
        "openff-2.2.0.offxml"
    )
    protocol_settings.thermo_settings.temperature = 298.15 * OFFUnit.kelvin
    protocol_settings.thermo_settings.pressure = 1 * OFFUnit.bar
    protocol_settings.solvation_settings.box_shape = "dodecahedron"
    protocol_settings.alchemical_settings.softcore_LJ = "gapsys"
    protocol_settings.simulation_settings.equilibration_length = (
        1.0 * OFFUnit.nanoseconds
    )
    protocol_settings.simulation_settings.production_length = 5.0 * OFFUnit.nanoseconds
    protocol_settings.simulation_settings.time_per_iteration = 1 * OFFUnit.picoseconds
    protocol_settings.protocol_repeats = 1

    return protocol_settings


def _asap_septop_settings() -> "Settings":
    """ASAP-tuned default settings for ``SepTopProtocol``.

    Applies the same force-field and thermodynamic conditions as the RFE defaults
    (openff-2.2.0, 298.15 K / 1 bar) and limits to a single protocol repeat, while
    leaving simulation lengths and lambda schedules at the openfe upstream defaults.
    """
    from openff.units import unit as OFFUnit

    protocol_class = get_protocol_class("SepTopProtocol")
    protocol_settings = protocol_class.default_settings().unfrozen_copy()

    protocol_settings.forcefield_settings.small_molecule_forcefield = (
        "openff-2.2.0.offxml"
    )
    protocol_settings.thermo_settings.temperature = 298.15 * OFFUnit.kelvin
    protocol_settings.thermo_settings.pressure = 1 * OFFUnit.bar
    protocol_settings.protocol_repeats = 1

    return protocol_settings


def _asap_absolute_binding_settings() -> "Settings":
    """ASAP-tuned default settings for ``AbsoluteBindingProtocol``.

    Applies the same force-field and thermodynamic conditions as the RFE defaults
    and limits to a single protocol repeat. Simulation lengths and lambda schedules
    are left at the openfe upstream defaults.
    """
    from openff.units import unit as OFFUnit

    protocol_class = get_protocol_class("AbsoluteBindingProtocol")
    protocol_settings = protocol_class.default_settings().unfrozen_copy()

    protocol_settings.forcefield_settings.small_molecule_forcefield = (
        "openff-2.2.0.offxml"
    )
    protocol_settings.thermo_settings.temperature = 298.15 * OFFUnit.kelvin
    protocol_settings.thermo_settings.pressure = 1 * OFFUnit.bar
    protocol_settings.protocol_repeats = 1

    return protocol_settings

def _asap_fah_nonequilibrium_cycling_settings() -> "Settings":
    """ASAP-tuned default settings for ``FahNonEquilibriumCyclingProtocol``.

    Sets compute platform explicitly to ``None`` to avoid e.g. ``CUDA``
    failures on the work server.
    """
    protocol_class = get_protocol_class("FahNonEquilibriumCyclingProtocol")
    protocol_settings = protocol_class.default_settings().unfrozen_copy()

    # the default for this setting upstream is `CUDA`;
    # setting it to `None` will use the fastest platform available on the host,
    # and not raise an exception if a GPU is not present;
    # this setting has no bearing on `openm-core` behavior downstream
    protocol_settings.engine_settings.compute_platform = None

    return protocol_settings


#: Builders that produce the ASAP-Alchemy default settings for a protocol. Where a
#: protocol is absent, the protocol's own ``default_settings()`` is used unchanged.
_DEFAULT_SETTINGS_BUILDERS = {
    "RelativeHybridTopologyProtocol": _asap_relative_hybrid_topology_settings,
    "SepTopProtocol": _asap_septop_settings,
    "AbsoluteBindingProtocol": _asap_absolute_binding_settings,
    "FahNonEquilibriumCyclingProtocol": _asap_fah_nonequilibrium_cycling_settings,
}


def default_protocol_settings(name: str) -> "Settings":
    """Return the default ``Settings`` for the protocol registered under ``name``.

    For protocols with ASAP-specific defaults (e.g. ``RelativeHybridTopologyProtocol``)
    those tuned settings are returned; otherwise the protocol's own
    ``default_settings()`` is used.
    """
    if name in _DEFAULT_SETTINGS_BUILDERS:
        return _DEFAULT_SETTINGS_BUILDERS[name]()
    return get_protocol_class(name).default_settings()


def build_protocol(name: str, settings: "Settings") -> "gufe.Protocol":
    """Instantiate the protocol registered under ``name`` with the given ``settings``."""
    return get_protocol_class(name)(settings=settings)


def protocol_name_for(protocol: "gufe.Protocol") -> str:
    """Return the registered name for a protocol *object*.

    Used to map a built ``gufe.Protocol`` (e.g. attached to a ``Transformation``)
    back to its ASAP-Alchemy identifier so that results can be separated by
    protocol.

    Args:
        protocol: A protocol instance (or any object whose class name matches a
            registered protocol).

    Raises:
        ValueError: If the object's class is not a registered protocol.
    """
    class_name = type(protocol).__name__
    if class_name not in PROTOCOL_REGISTRY:
        raise ValueError(
            f"Protocol class {class_name!r} is not registered with ASAP-Alchemy; "
            f"registered protocols are {available_protocols()}."
        )
    return class_name
