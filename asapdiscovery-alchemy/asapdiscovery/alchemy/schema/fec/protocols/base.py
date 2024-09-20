import abc
import enum
from typing import Literal

from asapdiscovery.alchemy.schema.base import _SchemaBase


class SupportedProtocols(enum.Enum):
    RelativeHybridTopologyProtocol = "RelativeHybridTopologyProtocol"
    NonEquilibriumCyclingProtocol = "NonEquilibriumCyclingProtocol"


class _ProtocolSettingsBase(_SchemaBase, abc.ABC):
    """
    A base class for openfe protocol settings to work around the serialisation issues with
    openFE settings models see <https://github.com/OpenFreeEnergy/openfe/issues/518>.
    """

    type: Literal["_ProtocolSettingsBase"] = "_ProtocolSettingsBase"

    @classmethod
    def from_defaults(cls):
        """
        Our opinionated general settings which should be applied to all
        protocols to make certain settings consistent.
        """
        from openff.units import unit

        protocol_settings = cls._from_defaults()
        # Set the small molecule force field
        protocol_settings.forcefield_settings.small_molecule_forcefield = "openff-2.2.0"
        # Thermo settings
        protocol_settings.thermo_settings.temperature = 298.15 * unit.kelvin
        protocol_settings.thermo_settings.pressure = 1 * unit.bar
        return protocol_settings

    @classmethod
    @abc.abstractmethod
    def _from_defaults(cls): ...

    @abc.abstractmethod
    def to_openfe_protocol(self): ...
