from typing import Literal

from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe import (
    RelativeHybridTopologyProtocolSettings as RelativeHybridTopologyProtocolSettings_,
)

from .base import ProtocolSettingsBase


class RelativeHybridTopologySettings(
    RelativeHybridTopologyProtocolSettings_, ProtocolSettingsBase
):

    type: Literal["RelativeHybridTopologySettings"] = "RelativeHybridTopologySettings"

    @classmethod
    def from_defaults(cls):
        settings = RelativeHybridTopologyProtocol.default_settings()

        # NOTE: remove this if we want to just take the default from `openfe` as
        # it advances
        settings.forcefield_settings.small_molecule_forcefield = "openff-2.2.0"

        settings.protocol_repeats = 1

        return cls(**dict(settings))

    def to_openfe_protocol(self):
        settings = dict(self)
        settings.pop("type")

        protocol_settings = RelativeHybridTopologyProtocolSettings_(**settings)
        return RelativeHybridTopologyProtocol(settings=protocol_settings)
