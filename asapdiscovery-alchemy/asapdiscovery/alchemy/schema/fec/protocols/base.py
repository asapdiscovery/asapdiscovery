import abc
import enum
from typing import Literal

from ...base import _SchemaBase


class SupportedProtocols(enum.Enum):
    RelativeHybridTopologyProtocol = "RelativeHybridTopologyProtocol"
    NonEquilibriumCyclingProtocol = "NonEquilibriumCyclingProtocol"


class ProtocolSettingsBase(_SchemaBase, abc.ABC):
    """
    A base class for openfe protocol settings to work around the serialisation issues with
    openFE settings models see <https://github.com/OpenFreeEnergy/openfe/issues/518>.
    """

    type: Literal["_ProtocolBase"] = "_ProtocolBase"

    @classmethod
    @abc.abstractmethod
    def from_defaults(cls):
        ...

    @abc.abstractmethod
    def to_openfe_protocol(self): ...
