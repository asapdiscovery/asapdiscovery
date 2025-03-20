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
from openfe.setup.atom_mapping import lomap_scorers, perses_scorers
from openff.models.types import FloatQuantity
from openff.units import unit as OFFUnit
from pydantic.v1 import Field

from ._util import check_ligand_series_uniqueness_and_names
from .base import _SchemaBase, _SchemaBaseFrozen
from .network import NetworkPlanner, PlannedNetwork

if TYPE_CHECKING:
    from asapdiscovery.data.schema.ligand import Ligand
    from gufe.mapping import LigandAtomMapping


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


class AdaptiveSettings(_SchemaBase):
    """
    A settings class to encode settings for adaptive settings. These were recommended by OpenFE.
    """

    type: Literal["AdaptiveSettings"] = "AdaptiveSettings"
    adaptive_sampling: bool = Field(
        False,
        description="If True, will enable increase in production length of simulations given a `adaptive_sampling_multiplier` and `adaptive_sampling_threshold`.",
    )
    adaptive_sampling_multiplier: float = Field(
        2.0,
        description="The number of times more production simulation length (sampling time) that will be assigned to edges whose mapping scoring falls below the `adaptive_sampling_threshold`.",
    )
    adaptive_sampling_threshold: float = Field(
        0.5,
        description="The threshold that separates edges that are expected to perform well (higher; regular production simulation time) and poorly (lower; regular production simulation time * `adaptive_sampling_multiplier`). Recommended settings are 0.5 (LOMAP scorer) or 0.85 (PERSES scorer).",
    )
    adaptive_solvent_padding: bool = Field(
        True,
        description="Whether or not to use adaptive solvent padding; typically the complex phase can handle smaller padding size.",
    )
    solvent_padding_complex: FloatQuantity["nanometer"] = Field(  # noqa: F821
        1.5 * OFFUnit.nanometer,
        description="The solvent padding (in nm) to use for the complex phase of each edge.",
    )
    solvent_padding_solvated: FloatQuantity["nanometer"] = Field(  # noqa: F821
        1.5 * OFFUnit.nanometer,
        description="The solvent padding (in nm) to use for the solvated phase of each edge.",
    )

    def get_adapted_sampling_protocol(
        self,
        scorer_method: str,
        mapping: "LigandAtomMapping",
        protocol: openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol,
        base_sampling_length: FloatQuantity["nanometer"],  # noqa: F821
    ) -> openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol:
        """
        It's advisable to increase simulation time on edges that are expected to be less reliable. There
        Aren't many good estimators for this, but the network planner edge scoring is a decent approximation.

        If the edge scoring (computed using `scorer_method`) is below the `adaptive_sampling_threshold` the
        simulation time is multiplied by `adaptive_sampling_multiplier`. Just to be sure, we use the base
        protocol's sampling time and not the provided edge protocol sampling time as a base value.

        Returns the adjusted OpenFE Protocol.
        """
        if scorer_method == "default_lomap":
            scorer = lomap_scorers.default_lomap_score
        elif scorer_method == "default_perses":
            scorer = perses_scorers.default_perses_scorer
        else:
            raise ValueError(
                f"Atom mapping scorer {scorer_method} not recognized; use one of `default_lomap`, `default_perses`."
            )
        if scorer(mapping) < self.adaptive_sampling_threshold:
            protocol._settings.simulation_settings.production_length = (
                base_sampling_length * self.adaptive_sampling_multiplier
            )
        return protocol

    def get_adapted_solvent_protocol(
        self,
        leg: str,
        protocol: openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol,
    ) -> openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol:
        """
        Certain water box shapes (such as dodecahedron) are able to handle slightly smaller padding size
        in the complex phase compared to the solvated phase. Given the leg (either "solvent" or "complex")
        this method applies the specified padding per phase (`solvent_padding_solvated` or
        `solvent_padding_complex`, resp.).

        Returns the adjusted OpenFE Protocol.
        """
        if leg == "solvent":
            protocol._settings.solvation_settings.solvent_padding = (
                self.solvent_padding_solvated
            )
        else:
            protocol._settings.solvation_settings.solvent_padding = (
                self.solvent_padding_complex
            )
        return protocol

    def apply_settings(
        self,
        edge_protocol: openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol,
        network_scorer: str,
        mapping: "LigandAtomMapping",
        leg: str,
        base_protocol: openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol,
    ) -> openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol:
        """
        Applies a set of adaptive settings to an OpenFE Protocol if requested.
        """
        import copy

        # create a copy of the edge_protocol to make it editable - we're returning the copy
        edge_protocol = copy.deepcopy(edge_protocol)

        # double the simulation time if requested
        if self.adaptive_sampling:
            base_sampling_length = (
                base_protocol.settings.simulation_settings.production_length
            )
            edge_protocol = self.get_adapted_sampling_protocol(
                network_scorer, mapping, edge_protocol, base_sampling_length
            )

        # adjust solvent padding per phase if requested
        if self.adaptive_solvent_padding:
            edge_protocol = self.get_adapted_solvent_protocol(leg, edge_protocol)

        return edge_protocol


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
            small_molecule_forcefield="openff-2.2.0.offxml"
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
            time_per_iteration=1 * OFFUnit.picoseconds,
        ),
        description="Settings for simulation control, including lengths and writing to disk.",
    )
    adaptive_settings: Optional[AdaptiveSettings] = Field(
        AdaptiveSettings(),
        description="Run adaptive settings depending on e.g. expected edge reliability or system phase.",
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
        import copy

        transformations = []
        # do all openfe conversions
        ligand_network = self.network.to_ligand_network()
        solvent = self.solvent_settings.to_solvent_component()
        receptor = self.to_openfe_receptor()
        protocol = self.to_openfe_protocol()

        # build the network
        for mapping in ligand_network.edges:
            # returns the name of the ff if we have no bespoke parameters
            ff_string = self._inject_bespoke_parameters(edge=mapping)
            # make a copy of the protocol and add the bespoke force field
            edge_protocol = copy.deepcopy(protocol)
            # make the settings editable
            edge_protocol._settings = edge_protocol._settings.unfrozen_copy()
            edge_protocol._settings.forcefield_settings.small_molecule_forcefield = (
                ff_string
            )

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

                # run this edge's protocol through adaptive settings. If this list of things to pass
                # grows any larger we should only pass the `FreeEnergyCalculationNetwork` and instead
                # infer these parameters somewhere in self.adaptive_settings.
                if self.adaptive_settings:
                    edge_protocol = self.adaptive_settings.apply_settings(
                        edge_protocol,  # the protocol to be adjusted
                        self.network.scorer,  # the network edge scorer - for adaptive sampling
                        mapping,  # the atom mapping for this edge - for adaptive sampling
                        leg,  # whether this edge is complex or solvated phase - for adaptive solvent box padding
                        protocol,  # base protocol to compare with for internal checking
                    )

                # set up the transformation
                transformation = openfe.Transformation(
                    stateA=system_a,
                    stateB=system_b,
                    mapping={"ligand": mapping},
                    protocol=edge_protocol,  # use protocol created above
                    name=f"{system_a.name}_{system_b.name}",
                )
                transformations.append(transformation)

        return openfe.AlchemicalNetwork(edges=transformations, name=self.dataset_name)

    def _inject_bespoke_parameters(self, edge: "LigandAtomMapping") -> str:
        """
        Inject the bespoke torsion parameters for the given edge into the base force field.

        Args:
            edge: The edge from the OpenFE alchemical network which we want the parameters for.

        Returns:
            The string of the force field with bespoke parameters added or the name of the base force field if no
            bespoke parameters are found

        Notes:
            They will always be added in the order of the mapping (ligandA, ligandB)
        """
        from openff.toolkit import ForceField
        from openff.toolkit.utils.exceptions import DuplicateParameterError
        from openff.units import unit

        # get the name of the base ff and load it
        ff_string = self.forcefield_settings.small_molecule_forcefield
        if ".offxml" not in ff_string:
            ff_string += ".offxml"
        ff = ForceField(ff_string)

        # map the names to ligands to quickly find the parameters
        names_to_ligands = {
            ligand.compound_name: ligand for ligand in self.network.ligands
        }

        # torsion data to manually set the phase idivf and periodicity
        torsion_data = {
            "idivf1": 1.0,
            "idivf2": 1.0,
            "idivf3": 1.0,
            "idivf4": 1.0,
            "phase1": 0.0 * unit.degree,
            "phase2": 180 * unit.degree,
            "phase3": 0 * unit.degree,
            "phase4": 180 * unit.degree,
            "periodicity1": 1,
            "periodicity2": 2,
            "periodicity3": 3,
            "periodicity4": 4,
        }

        # track if we have any bespoke parameters
        bespoke_parameters = False
        for ofe_ligand in [edge.componentA, edge.componentB]:
            if (
                edge_ligand := names_to_ligands[ofe_ligand.name]
            ).bespoke_parameters is not None:
                bespoke_parameters = True
                for parameter in edge_ligand.bespoke_parameters.parameters:
                    handler = ff.get_parameter_handler(parameter.interaction)
                    parameter_data = {
                        key: value * getattr(unit, parameter.units)
                        for key, value in parameter.values.items()
                    }
                    parameter_data["smirks"] = parameter.smirks
                    parameter_data["id"] = f"bespokefit_{edge_ligand.compound_name}"
                    if parameter.interaction == "ProperTorsions":
                        parameter_data.update(torsion_data)
                    try:
                        # similar ligands will share parameters so make sure we don't add it twice
                        handler.add_parameter(parameter_kwargs=parameter_data)
                    except DuplicateParameterError:
                        continue

        # if we found bespoke parameters return the new force field
        if bespoke_parameters:
            ff_string = ff.to_string()

        return ff_string


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
