from typing import Literal

from gufe.settings import OpenMMSystemGeneratorFFSettings
from feflow.protocols import NonEquilibriumCyclingProtocol
from feflow.settings import (
    NonEquilibriumCyclingSettings as NonEquilibriumCyclingSettings_,
)

from .base import ProtocolSettingsBase


class NonEquilibriumCyclingSettings(
    NonEquilibriumCyclingSettings_, ProtocolSettingsBase
):

    type: Literal["NonEquilibriumCyclingSettings"] = "NonEquilibriumCyclingSettings"

    # NOTE: temporary override of feflow type hint missing
    # remove once merged: https://github.com/OpenFreeEnergy/feflow/pull/88
    forcefield_settings: OpenMMSystemGeneratorFFSettings

    @classmethod
    def from_defaults(cls):
        settings = NonEquilibriumCyclingProtocol.default_settings()
        return cls(**dict(settings))

    def to_openfe_protocol(self):
        settings = dict(self)
        settings.pop('type')

        protocol_settings = NonEquilibriumCyclingSettings_(settings)
        return NonEquilibriumCyclingProtocol(settings=protocol_settings)
