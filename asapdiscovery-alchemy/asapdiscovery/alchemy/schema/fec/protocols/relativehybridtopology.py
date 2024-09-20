from typing import Literal

from asapdiscovery.alchemy.schema.fec.protocols.base import _ProtocolSettingsBase
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe import (
    RelativeHybridTopologyProtocolSettings as RelativeHybridTopologyProtocolSettings_,
)
from openff.units import unit


class RelativeHybridTopologySettings(
    RelativeHybridTopologyProtocolSettings_, _ProtocolSettingsBase
):

    type: Literal["RelativeHybridTopologySettings"] = "RelativeHybridTopologySettings"

    @classmethod
    def _from_defaults(cls):
        settings = RelativeHybridTopologyProtocol.default_settings()

        # NOTE: remove this if we want to just take the default from `openfe` as
        # it advances
        # set some of our preferred settings
        # only run the calculation once per dag, repeats are done via separate tasks in alchemiscale
        settings.protocol_repeats = 1
        # make sure the runtime settings are not changed
        settings.simulation_settings.equilibration_length = 1.0 * unit.nanosecond
        settings.simulation_settings.production_length = 5.0 * unit.nanosecond

        return cls(**dict(settings))

    def to_openfe_protocol(self):
        protocol_settings = RelativeHybridTopologyProtocolSettings_(
            **self.dict(exclude={"type"})
        )
        return RelativeHybridTopologyProtocol(settings=protocol_settings)
