from typing import Literal

from feflow.protocols import NonEquilibriumCyclingProtocol
from feflow.settings import NonEquilibriumCyclingSettings as NonEquilibriumCyclingSettings_

from .base import ProtocolSettingsBase


class NonEquilibriumCyclingSettings(NonEquilibriumCyclingSettings_, ProtocolSettingsBase):

    type: Literal["NonEquilibriumCyclingSettings"] = "NonEquilibriumCyclingSettings"

    def from_defaults(cls):
        settings = NonEquilibriumCyclingProtocol.default_settings()

        return cls(**dict(settings))

    def to_openfe_protocol(self):
        protocol_settings = NonEquilibriumCyclingSettings_(**dict(self))
        return NonEquilibriumCyclingProtocol(settings=protocol_settings)
