from typing import Literal

from asapdiscovery.alchemy.schema.fec.protocols.base import _ProtocolSettingsBase
from feflow.protocols import NonEquilibriumCyclingProtocol
from feflow.settings import (
    NonEquilibriumCyclingSettings as NonEquilibriumCyclingSettings_,
)
from gufe.settings import OpenMMSystemGeneratorFFSettings


class NonEquilibriumCyclingSettings(
    NonEquilibriumCyclingSettings_, _ProtocolSettingsBase
):

    type: Literal["NonEquilibriumCyclingSettings"] = "NonEquilibriumCyclingSettings"

    # NOTE: temporary override of feflow type hint missing
    # remove once merged: https://github.com/OpenFreeEnergy/feflow/pull/88
    forcefield_settings: OpenMMSystemGeneratorFFSettings

    @classmethod
    def _from_defaults(cls):
        settings = NonEquilibriumCyclingProtocol.default_settings()
        return cls(**dict(settings))

    def to_openfe_protocol(self):
        protocol_settings = NonEquilibriumCyclingSettings_(
            **self.dict(exclude={"type"})
        )
        return NonEquilibriumCyclingProtocol(settings=protocol_settings)
