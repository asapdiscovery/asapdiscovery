import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional

import gufe
import openfe
from alchemiscale import ScopedKey
from gufe import settings
from gufe.tokenization import GufeKey
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    AlchemicalSettings,
    LambdaSettings,
    MultiStateOutputSettings,
    OpenFFPartialChargeSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    MultiStateSimulationSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)
from openfe.setup.atom_mapping import (
    lomap_scorers,
    perses_scorers,
)
from openff.models.types import FloatQuantity
from openff.units import unit as OFFUnit
from pydantic import BaseSettings, Field

from ._util import check_ligand_series_uniqueness_and_names
from .base import _SchemaBase, _SchemaBaseFrozen
from .network import NetworkPlanner, PlannedNetwork

if TYPE_CHECKING:
    from asapdiscovery.data.schema.ligand import Ligand


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
    ALCHEMISCALE_URL: str = Field(
        "https://api.alchemiscale.org",
        description="The address of the alchemiscale instance to connect to.",
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
        description="If the net charge of the chemical system should be neutralized by the ions defined by `positive_ion` and `negative_ion`.",
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

    def name(self):
        """Make a name for this transformation based on the names of the ligands."""
        return "-".join([self.ligand_a, self.ligand_b])


class _BaseResults(_SchemaBaseFrozen):
    """
    A base results class which handles the collecting and processing of the results.
    """

    type: Literal["_BaseResults"] = "_BaseResults"
    results: list[TransformationResult] = Field(
        [], description="The list of results collected for this dataset."
    )

    def to_cinnabar_measurements(self):
        """
        For the given set of results combine the solvent and complex phases to make a list of cinnabar RelativeMeasurements

        Returns:
            A list of RelativeMeasurements made from the combined solvent and complex phases.
        """
        from collections import defaultdict

        import numpy as np
        from cinnabar import Measurement

        raw_results = defaultdict(list)
        # gather by transform
        for result in self.results:
            raw_results[result.name()].append(result)

        # make sure we have a solvent and complex phase for each result
        ligands_to_remove = []
        for name, transforms in raw_results.items():
            missing_phase = {"complex", "solvent"} - {t.phase for t in transforms}
            if missing_phase:
                warnings.warn(
                    f"The transformation {name} is missing simulated legs in the following phases {missing_phase}; removing"
                )
                ligands_to_remove.append(name)
            if len(transforms) > 2:
                # We have too many simulations for this transform
                raise RuntimeError(
                    f"The transformation {name} has too many simulated legs, found the following phases {[t.phase for t in transforms]} expected complex and solvent."
                )

        for name in ligands_to_remove:
            raw_results.pop(name)

        # make the cinnabar data
        all_results = []
        for leg1, leg2 in raw_results.values():
            complex_leg: TransformationResult = (
                leg1 if leg1.phase == "complex" else leg2
            )
            solvent_leg: TransformationResult = (
                leg1 if leg1.phase == "solvent" else leg2
            )
            result = Measurement(
                labelA=leg1.ligand_a,
                labelB=leg1.ligand_b,
                DG=(complex_leg.estimate - solvent_leg.estimate),
                # propagate errors
                uncertainty=np.sqrt(
                    complex_leg.uncertainty**2 + solvent_leg.uncertainty**2
                ),
                computational=True,
                source="calculated",
            )
            all_results.append(result)
        return all_results

    def to_fe_map(self):
        """
        Convert the set of relative free energy estimates to a cinnabar FEMap object to calculate the absolute values
        or plot vs experiment.

        Returns:
            A cinnabar.FEMap made from the relative results objects.
        """
        from cinnabar import FEMap

        fe_graph = FEMap()
        for result in self.to_cinnabar_measurements():
            fe_graph.add_measurement(measurement=result)
        return fe_graph


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
        settings.OpenMMSystemGeneratorFFSettings(
            small_molecule_forcefield="openff-2.2.0",
        ),
        description="The force field settings used to parameterize the systems.",
    )
    thermo_settings: settings.ThermoSettings = Field(
        settings.ThermoSettings(
            temperature=298.15 * OFFUnit.kelvin, pressure=1 * OFFUnit.bar
        ),
        description="The settings for thermodynamic parameters.",
    )
    solvation_settings: OpenMMSolvationSettings = Field(
        OpenMMSolvationSettings(box_shape="dodecahedron"),
        description="Settings controlling how the systems should be solvated using OpenMM.",
    )
    alchemical_settings: AlchemicalSettings = Field(
        AlchemicalSettings(softcore_LJ="gapsys"),
        description="The alchemical protocol settings.",
    )
    engine_settings: OpenMMEngineSettings = Field(
        OpenMMEngineSettings(), description="Openmm platform settings."
    )
    integrator_settings: IntegratorSettings = Field(
        IntegratorSettings(),
        description="Settings for the LangevinSplittingDynamicsMove integrator.",
    )
    simulation_settings: MultiStateSimulationSettings = Field(
        MultiStateSimulationSettings(
            equilibration_length=1.0 * OFFUnit.nanoseconds,
            production_length=5.0 * OFFUnit.nanoseconds,
        ),
        description="Settings for simulation control, including lengths and writing to disk.",
    )
    protocol: Literal["RelativeHybridTopologyProtocol"] = Field(
        "RelativeHybridTopologyProtocol",
        description="The name of the OpenFE alchemical protocol to use.",
    )
    protocol_repeats: int = Field(
        1,
        description="The number of extra times the calculation should be run and the results should be averaged over. Where 2 would mean run the calculation a total of 3 times.",
    )
    lambda_settings: LambdaSettings = Field(
        LambdaSettings(), description="Lambda schedule settings."
    )

    partial_charge_settings: OpenFFPartialChargeSettings = Field(
        OpenFFPartialChargeSettings(),
        description="The method which should be used to generate the partial charges if not provided with the ligand.",
    )
    output_settings: MultiStateOutputSettings = Field(
        MultiStateOutputSettings(),
        description="Settings for MultiState simulation output settings like writing to disk.",
    )

    def to_openfe_protocol(self):
        protocol_settings = openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocolSettings(
            # workaround type hint being base FF engine class
            forcefield_settings=self.forcefield_settings,
            thermo_settings=self.thermo_settings,
            # system_settings=self.system_settings,
            solvation_settings=self.solvation_settings,
            alchemical_settings=self.alchemical_settings,
            # alchemical_sampler_settings=self.alchemical_sampler_settings,
            engine_settings=self.engine_settings,
            integrator_settings=self.integrator_settings,
            simulation_settings=self.simulation_settings,
            lambda_settings=self.lambda_settings,
            protocol_repeats=self.protocol_repeats,
            partial_charge_settings=self.partial_charge_settings,
            output_settings=self.output_settings,
        )
        return openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol(
            settings=protocol_settings
        )


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
    experimental_protocol: Optional[str] = Field(
        None,
        description="The name of the experimental protocol in the CDD vault that should be associated with this Alchemy network.",
    )
    target: Optional[str] = Field(
        None,
        description="The name of the biological target associated with this Alchemy network.",
    )

    class Config:
        """Overwrite the class config to freeze the results model"""

        allow_mutation = False
        orm_mode = True

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

        # get the atom mapping scorer in case we need to double simulation time for potentially challenging edges
        if self.network.scorer == "default_lomap":
            scorer = lomap_scorers.default_lomap_score
            scorer_threshold = 0.5  # this is industry standard
        elif self.network.scorer == "default_perses":
            scorer = perses_scorers.default_perses_scorer
            scorer_threshold = 0.91  # recommended by OpenFE devs
        else:
            raise ValueError(
                f"Atom mapping scorer {self.network.scorer} not recognized; use one of `default_lomap`, `default_perses`."
            )
        adaptive_sampling = True  ##### TMP FOR DEV
        # build the network
        for mapping in ligand_network.edges:
            print()
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

                # make the OpenFE protocol for this edge; double the simulation time if requested
                protocol_copy = self.copy()
                if adaptive_sampling and scorer(mapping) < scorer_threshold:
                    protocol_copy.simulation_settings.production_length *= 2
                protocol_openfe = protocol_copy.to_openfe_protocol()
                print(protocol_copy.simulation_settings.production_length)

                # set up the transformation
                transformation = openfe.Transformation(
                    stateA=system_a,
                    stateB=system_b,
                    mapping={"ligand": mapping},
                    protocol=protocol_openfe,  # use protocol created above
                    name=f"{system_a.name}_{system_b.name}",
                )
                transformations.append(transformation)

        import sys

        sys.exit()
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
        ligands: Optional[list["Ligand"]] = None,
        central_ligand: Optional["Ligand"] = None,
        graphml: Optional[str] = None,
        experimental_protocol: Optional[str] = None,
        target: Optional[str] = None,
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
            experimental_protocol: The name of the experimental protocol in the CDD vault that should be
                associated with this Alchemy network.
            target: The name of the biological target associated with this Alchemy network.

         Returns:
             The planned FEC network which can be executed locally or submitted to alchemiscale.
        """
        # generate the network
        if ligands:
            check_ligand_series_uniqueness_and_names(ligands)
            # start by trying to plan the network
            planned_network = self.network_planner.generate_network(
                ligands=ligands,
                central_ligand=central_ligand,
            )
        # pre-generated network
        elif graphml:
            # equivalent name checks in constructor
            planned_network = PlannedNetwork.from_graphml(graphml)

        else:
            raise ValueError("Either ligands or a graphml file must be provided.")

        planned_fec_network = FreeEnergyCalculationNetwork(
            dataset_name=dataset_name,
            network=planned_network,
            receptor=receptor.to_json(),
            experimental_protocol=experimental_protocol,
            target=target,
            **self.dict(exclude={"type", "network_planner"}),
        )
        return planned_fec_network


class _BaseFailure(_SchemaBaseFrozen):
    """Base class for collecting errors and tracebacks from failed FEC runs"""

    type: Literal["_BaseFailure"] = "_BaseFailure"

    error: tuple[str, tuple[Any, ...]] = Field(
        tuple(), description="Exception raised and associated message."
    )
    traceback: str = Field(
        "", description="Complete traceback associated with the failure."
    )


class AlchemiscaleFailure(_BaseFailure):
    """Class for collecting errors and tracebacks from errored tasks in an alchemiscale network"""

    type: Literal["AlchemiscaleFailure"] = "AlchemiscaleFailure"

    network_key: ScopedKey = Field(
        ...,
        description="The alchemiscale key associated with this submitted network, which is used to gather the failed results from the client.",
    )
    task_key: ScopedKey = Field(..., description="Task key for the errored task.")
    unit_key: GufeKey = Field(
        ..., description="Protocol unit key associated to the errored task."
    )
    dag_result_key: GufeKey = Field(
        ..., description="Protocol DAG result key associated to the errored task."
    )
