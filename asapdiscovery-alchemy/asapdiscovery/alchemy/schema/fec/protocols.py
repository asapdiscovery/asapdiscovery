import abc
from typing import Literal

import gufe
import openfe.protocols.openmm_rfe
from asapdiscovery.alchemy.schema.fec.base import _SchemaBase
from feflow.settings import PeriodicNonequilibriumIntegratorSettings
from gufe import settings
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    AlchemicalSamplerSettings,
    AlchemicalSettings,
    IntegratorSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
    SimulationSettings,
    SystemSettings,
)
from openff.models.types import FloatQuantity
from openff.units import unit as OFFUnit
from pydantic import Field


# DD: I think we'll want to try and avoid duplicating the attribute structure of Protocol settings here;
# though we don't expect them to change often, it would be cleaner to avoid duplication
class _ProtocolBase(_SchemaBase, abc.ABC):
    """
    A base class for openfe protocol settings to work around the serialisation issues with
    openFE settings models see <https://github.com/OpenFreeEnergy/openfe/issues/518>.
    """

    type: Literal["_ProtocolBase"] = "_ProtocolBase"

    @abc.abstractmethod
    def to_openfe_protocol(self): ...


# DD TODO: extract defaults from below and make sure these make it into our implementations


class RelativeHybridTopologySettings(_ProtocolBase):
    """A settings class for the RelativeHybridTopologyProtocol in OpenFE."""

    type: Literal["RelativeHybridTopologySettings"] = "RelativeHybridTopologySettings"

    simulation_settings: SimulationSettings = Field(
        SimulationSettings(
            equilibration_length=1.0 * OFFUnit.nanoseconds,
            production_length=5.0 * OFFUnit.nanoseconds,
        ),
        description="Settings for simulation control, including lengths and writing to disk.",
    )
    n_repeats: int = Field(
        2,
        description="The number of extra times the calculation should be run and the results should be averaged over. Where 2 would mean run the calculation a total of 3 times.",
    )
    # note alchemical_sampler_settings.n_repeats specifies the number of times each transformation will be run
    alchemical_sampler_settings: AlchemicalSamplerSettings = Field(
        AlchemicalSamplerSettings(
            n_repeats=1
        ),  # Run one calculation in serial and parallise accross alchemiscale workers see n_repeats on the _FreeEnergyBase object
        description="Settings for the Equilibrium Alchemical sampler, currently supporting either MultistateSampler, SAMSSampler or ReplicaExchangeSampler.",
    )

    def to_openfe_protocol(self):
        protocol_settings = openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocolSettings(
            # workaround type hint being base FF engine class
            forcefield_settings=self.forcefield_settings,
            thermo_settings=self.thermo_settings,
            system_settings=self.system_settings,
            solvation_settings=self.solvation_settings,
            alchemical_settings=self.alchemical_settings,
            alchemical_sampler_settings=self.alchemical_sampler_settings,
            engine_settings=self.engine_settings,
            integrator_settings=self.integrator_settings,
            simulation_settings=self.simulation_settings,
        )
        return openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol(
            settings=protocol_settings
        )


class NonEquilibriumCyclingSettings(_ProtocolBase):
    """A settings class for the NonEquilibriumCycling protocol on feflow"""

    type: Literal["NonEquilibriumCyclingSettings"] = "NonEquilibriumCyclingSettings"

    integrator_settings: PeriodicNonequilibriumIntegratorSettings = Field(
        PeriodicNonequilibriumIntegratorSettings(),
        description="Settings for the periodic non-equilibrium integrator.",
    )
    num_cycles: int = Field(
        100,
        description="The number of non-equilibrium cycles to run for this protocol.",
    )

    def to_openfe_protocol(self):
        from feflow.protocols import NonEquilibriumCyclingProtocol
        from feflow.settings import NonEquilibriumCyclingSettings

        protocol_settings = NonEquilibriumCyclingSettings()
        return NonEquilibriumCyclingProtocol(settings=protocol_settings)


OPENFE_PROTOCOLS = [RelativeHybridTopologySettings, NonEquilibriumCyclingSettings]
