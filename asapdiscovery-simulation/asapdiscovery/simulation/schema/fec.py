from typing import Literal, Optional

import gufe
import openfe
from alchemiscale import ScopedKey
from asapdiscovery.simulation.schema.base import _SchemaBase, _SchemaBaseFrozen
from asapdiscovery.simulation.schema.network import NetworkPlanner, PlannedNetwork
from gufe import settings
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    AlchemicalSamplerSettings,
    AlchemicalSettings,
    IntegratorSettings,
    OpenMMEngineSettings,
    SimulationSettings,
    SolvationSettings,
    SystemSettings,
)
from openff.models.types import FloatQuantity
from openff.units import unit as OFFUnit
from pydantic import BaseSettings, Field


class AlchemiscaleSettings(BaseSettings):
    """
    General settings class to capture Alchemiscale credentials from the environment.
    """

    ALCHEMISCALE_ID: str = Field(
        ..., description="Your personal alchemiscale ID used to login."
    )
    ALCHEMISCALE_KEY: str = Field(
        ..., description="Your personal alchemiscale Key used to login."
    )


class SolventSettings(_SchemaBase):
    """
    A settings class to encode the solvent used in the OpenFE FEC calculations.
    """

    type: Literal["SolventSettings"] = "SolventSettings"

    smiles: str = Field("O", description="The smiles pattern of the solvent.")
    positive_ion: str = Field(
        "Na+",
        description="The positive monoatomic ion which should be used to neutralize the system and to adjust the ionic concentration.",
    )
    negative_ion: str = Field(
        "Cl-",
        description="The negative monoatomic ion which should be used to neutralize the system and to adjust the ionic concentration.",
    )
    neutralize: bool = Field(
        True,
        description="If the net charge of the chemical system should be neutralized by the ions defined by `positive_ion` abd `negative_ion`.",
    )
    ion_concentration: FloatQuantity["molar"] = Field(  # noqa: F821
        0.15 * OFFUnit.molar,
        description="The ionic concentration required in molar units.",
    )

    def to_solvent_component(self) -> gufe.SolventComponent:
        return gufe.SolventComponent(**self.dict(exclude={"type"}))


# TODO make base class with abstract methods to collect results.
class TransformationResult(_SchemaBaseFrozen):
    """
    Store the results of a transformation, note when retries are used this will be the average result.
    """

    type: Literal["TransformationResult"] = "TransformationResult"
    ligand_a: str = Field(
        ..., description="The name of the ligand in state A of the transformation."
    )
    ligand_b: str = Field(
        ..., description="The name of the ligand in state B of the transformation."
    )
    phase: Literal["complex", "solvent"] = Field(
        ..., description="The phase of the transformation."
    )
    estimate: FloatQuantity["kcal/mol"] = Field(  # noqa: F821
        ..., description="The average estimate of this transformation in kcal/mol"
    )
    uncertainty: FloatQuantity["kcal/mol"] = Field(  # noqa: F821
        ...,
        description="The standard deviation of the estimates of this transform in kcal/mol",
    )


class _BaseResults(_SchemaBaseFrozen):
    """
    A base results class which handles the collecting and processing of the results.
    """

    type: Literal["_BaseResults"] = "_BaseResults"
    results: list[TransformationResult] = Field(
        [], description="The list of results collected for this dataset."
    )

    def to_cinnabar_csv(self, file_name: str):
        """Create a csv file which can be read by cinnabar for analysis."""
        pass


class AlchemiscaleResults(_BaseResults):
    type: Literal["AlchemiscaleResults"] = "AlchemiscaleResults"

    network_key: ScopedKey = Field(
        ...,
        description="The alchemiscale key associated with this submited network, which is used to gather results from the client.",
    )


class _FreeEnergyBase(_SchemaBase):
    """
    A base class for the FreeEnergyCalculationFactory and Network to work around the serialisation issues with
    openFE settings models see <https://github.com/OpenFreeEnergy/openfe/issues/518>.
    """

    type: Literal["_FreeEnergyBase"] = "_FreeEnergyBase"

    solvent_settings: SolventSettings = Field(
        SolventSettings(),
        description="The solvent settings which should be used during the free energy calculations.",
    )
    forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = Field(
        settings.OpenMMSystemGeneratorFFSettings(),
        description="The force field settings used to parameterize the systems.",
    )
    thermo_settings: settings.ThermoSettings = Field(
        settings.ThermoSettings(
            temperature=298.15 * OFFUnit.kelvin, pressure=1 * OFFUnit.bar
        ),
        description="The settings for thermodynamic parameters.",
    )
    system_settings: SystemSettings = Field(
        SystemSettings(), description="The nonbonded system settings."
    )
    solvation_settings: SolvationSettings = Field(
        SolvationSettings(),
        description="Settings controlling how the systems should be solvated.",
    )
    alchemical_settings: AlchemicalSettings = Field(
        AlchemicalSettings(), description="The alchemical protocol settings."
    )
    alchemical_sampler_settings: AlchemicalSamplerSettings = Field(
        AlchemicalSamplerSettings(),
        description="Settings for the Equilibrium Alchemical sampler, currently supporting either MultistateSampler, SAMSSampler or ReplicaExchangeSampler.",
    )
    engine_settings: OpenMMEngineSettings = Field(
        OpenMMEngineSettings(), description="Openmm platform settings."
    )
    integrator_settings: IntegratorSettings = Field(
        IntegratorSettings(),
        description="Settings for the LangevinSplittingDynamicsMove integrator.",
    )
    simulation_settings: SimulationSettings = Field(
        SimulationSettings(
            equilibration_length=1.0 * OFFUnit.nanoseconds,
            production_length=5.0 * OFFUnit.nanoseconds,
        ),
        description="Settings for simulation control, including lengths and writing to disk.",
    )
    protocol: Literal["RelativeHybridTopologyProtocol"] = Field(
        "RelativeHybridTopologyProtocol",
        description="The name of the OpenFE alchemical protocol to use.",
    )
    n_repeats: int = Field(
        2,
        description="The number of extra times the calculation should be run and the results should be averaged over. Where 2 would mean run the calculation a total of 3 times.",
    )

    def to_openfe_protocol(self):
        """Build the corresponding OpenFE protocol from the settings defined in this schema."""
        # TODO we need some way to link the settings to the protocol for when we have other options
        if self.protocol == "RelativeHybridTopologyProtocol":
            protocol_class = openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol
            settings_class = (
                openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocolSettings
            )

        protocol_settings = settings_class(
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
        return protocol_class(settings=protocol_settings)


class FreeEnergyCalculationNetwork(_FreeEnergyBase):
    """
    A schema of a FEC network created by the FreeEnergyCalculationFactory which contains all runtime settings and can
    be converted to local openFE inputs or submitted to alchemiscale.
    """

    type: Literal["FreeEnergyCalculationNetwork"] = "FreeEnergyCalculationNetwork"
    dataset_name: str = Field(
        ...,
        description="The name of the dataset, this will be used for local files and the alchemiscale network.",
    )
    network: PlannedNetwork = Field(
        ...,
        description="The planned free energy network with atom mappings between ligands.",
    )
    receptor: str = Field(
        ...,
        description="The JSON str of the receptor which should be used in the FEC calculation.",
    )
    results: Optional[AlchemiscaleResults] = Field(
        None,
        description="The results object which tracks how the calculation was run locally or on alchemiscale and stores the physical results.",
    )

    class Config:
        """Overwrite the class config to freeze the results model"""

        allow_mutation = False

    def to_openfe_receptor(self) -> openfe.ProteinComponent:
        return openfe.ProteinComponent.from_json(self.receptor)

    def to_alchemical_network(self) -> openfe.AlchemicalNetwork:
        """
        Create an openfe AlchemicalNetwork from the planned network which can be submitted to alchemiscale or ran locally

        Returns:
            An openfe.AlchemicalNetwork created from the schema.
        """
        transformations = []
        # do all openfe conversions
        ligand_network = self.network.to_ligand_network()
        solvent = self.solvent_settings.to_solvent_component()
        receptor = self.to_openfe_receptor()
        protocol = self.to_openfe_protocol()

        # build the network
        for mapping in ligand_network.edges:
            for leg in ["solvent", "complex"]:
                sys_a_dict = {"ligand": mapping.componentA, "solvent": solvent}
                sys_b_dict = {"ligand": mapping.componentB, "solvent": solvent}

                if leg == "complex":
                    sys_a_dict["protein"] = receptor
                    sys_b_dict["protein"] = receptor

                system_a = openfe.ChemicalSystem(
                    sys_a_dict, name=f"{mapping.componentA.name}_{leg}"
                )
                system_b = openfe.ChemicalSystem(
                    sys_b_dict, name=f"{mapping.componentB.name}_{leg}"
                )

                transformation = openfe.Transformation(
                    stateA=system_a,
                    stateB=system_b,
                    mapping={"ligand": mapping},
                    protocol=protocol,  # use protocol created above
                    name=f"{self.dataset_name}{system_a.name}_{system_b.name}",
                )
                transformations.append(transformation)

        return openfe.AlchemicalNetwork(edges=transformations, name=self.dataset_name)


class FreeEnergyCalculationFactory(_FreeEnergyBase):
    """A factory class to configure FEC calculations using the openFE pipeline. This generates a prepared FEC network
    which can be executed locally or submitted to Alchemiscale."""

    type: Literal["FreeEnergyCalculationFactory"] = "FreeEnergyCalculationFactory"

    network_planner: NetworkPlanner = Field(
        NetworkPlanner(),
        description="The network planner settings which should be used to construct the network.",
    )

    def create_fec_dataset(
        self,
        dataset_name: str,
        receptor: openfe.ProteinComponent,
        ligands: list[openfe.SmallMoleculeComponent],
        central_ligand: Optional[openfe.SmallMoleculeComponent] = None,
    ) -> FreeEnergyCalculationNetwork:
        """
         Use the factory settings to create a FEC dataset using OpenFE models.

        Args:
            dataset_name: The name which should be given to this dataset, this will be used for local file creation or
            to identify on alchemiscale
            receptor: The prepared receptor to use in the FEC dataset.
            ligands: The list of prepared and state enumerated ligands to use in the FEC calculation.
            central_ligand: An optional ligand which should be considered as the center only needed for radial networks.
            Note this ligand will be deduplicated from the list if it appears in both.

         Returns:
             The planned FEC network which can be executed locally or submitted to alchemiscale.
        """

        # start by trying to plan the network
        planned_network = self.network_planner.generate_network(
            ligands=ligands, central_ligand=central_ligand
        )

        planned_fec_network = FreeEnergyCalculationNetwork(
            dataset_name=dataset_name,
            network=planned_network,
            receptor=receptor.to_json(),
            **self.dict(exclude={"type", "network_planner"}),
        )
        return planned_fec_network
