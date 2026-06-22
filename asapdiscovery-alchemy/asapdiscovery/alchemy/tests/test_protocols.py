"""Tests for the alchemical protocol registry."""

import gufe
import pytest

from asapdiscovery.alchemy.schema import protocols as protocol_registry
from asapdiscovery.alchemy.schema.protocols import (
    available_protocols,
    build_protocol,
    default_protocol_settings,
    get_protocol_class,
    protocol_name_for,
)


def test_available_protocols():
    names = available_protocols()
    assert "RelativeHybridTopologyProtocol" in names
    assert "NonEquilibriumCyclingProtocol" in names
    assert "FahNonEquilibriumCyclingProtocol" in names


@pytest.mark.parametrize("name", available_protocols())
def test_build_and_roundtrip_name(name):
    """Each registered protocol can be built and mapped back to its name."""
    settings = default_protocol_settings(name)
    protocol = build_protocol(name, settings)
    assert isinstance(protocol, gufe.Protocol)
    assert protocol_name_for(protocol) == name


def test_unknown_protocol_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown protocol"):
        get_protocol_class("NotARealProtocol")


def test_protocol_name_for_unregistered_raises():
    class _NotRegistered:
        pass

    with pytest.raises(ValueError, match="not registered"):
        protocol_name_for(_NotRegistered())


def test_asap_rfe_defaults_are_preserved():
    """The RFE defaults reproduce ASAP's historical tuned settings, not raw openfe."""
    settings = default_protocol_settings("RelativeHybridTopologyProtocol")
    assert (
        settings.forcefield_settings.small_molecule_forcefield == "openff-2.2.0.offxml"
    )
    assert settings.solvation_settings.box_shape == "dodecahedron"
    assert settings.alchemical_settings.softcore_LJ == "gapsys"
    assert settings.protocol_repeats == 1


def test_missing_package_raises_importerror(monkeypatch):
    """A clear ImportError is raised when a protocol's package is unavailable."""
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "feflow.protocols":
            raise ImportError("No module named 'feflow'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(
        protocol_registry.importlib, "import_module", fake_import_module
    )

    with pytest.raises(ImportError, match="feflow"):
        get_protocol_class("NonEquilibriumCyclingProtocol")
