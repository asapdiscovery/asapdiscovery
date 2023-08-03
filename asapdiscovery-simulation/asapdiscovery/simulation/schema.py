import abc
from collections import namedtuple
from typing import Literal, Optional, Union

import gufe
import openfe
from gufe import settings
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    AlchemicalSamplerSettings,
    AlchemicalSettings,
    IntegratorSettings,
    OpenMMEngineSettings,
    RelativeHybridTopologyProtocolSettings,
    SimulationSettings,
    SolvationSettings,
    SystemSettings,
)
from openff.models.models import DefaultModel
from openff.units import unit as OFFUnit
from openmm.app import PME, HBonds
from openmm.unit import amu, nanometers
from pydantic import Field, PositiveFloat

ForceFieldParams = namedtuple(
    "ForceFieldParams",
    [
        "ff_xmls",
        "padding",
        "water_model",
        "nonbonded_method",
        "nonbonded_cutoff",
        "ewald_error_tolerance",
        "constraints",
        "rigid_water",
        "hydrogen_mass",
    ],
)


DefaultForceFieldParams = ForceFieldParams(
    ff_xmls=[
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ],
    padding=0.9 * nanometers,
    water_model="tip3p",
    nonbonded_method=PME,
    nonbonded_cutoff=1.0 * nanometers,
    ewald_error_tolerance=0.00001,
    constraints=HBonds,
    rigid_water=True,
    hydrogen_mass=4.0 * amu,
)


class _SchemaBase(abc.ABC, DefaultModel):
    """
    A basic schema class used to define the components of the Free energy workflow
    """

    type: Literal["base"] = "base"

    def to_file(self, filename: str):
        """
        Write the model to JSON file.
        """
        with open(filename, "w") as output:
            output.write(self.json(indent=2))

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the model from a JSON file
        """
        return cls.parse_file(filename)


class _BaseAtomMapper(_SchemaBase):
    """
    A base atom mapper which should be used to configure the method used to generate atom mappings.
    """

    @abc.abstractmethod
    def _get_mapper(self):
        ...

    def get_mapper(self):
        return self._get_mapper()

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        """
        Gather the names and versions of the Software used to calculate the atom mapping.
        """
        ...


class LomapAtomMapper(_BaseAtomMapper):
    """
    A settings class for the LomapAtomMapper in openFE
    """

    type: Literal["LomapAtomMapper"] = "LomapAtomMapper"

    time: PositiveFloat = Field(
        20, description="The timeout in seconds of the MCS algorithm pass to RDKit"
    )
    threed: bool = Field(
        True,
        description="If positional info should used to choose between symmetrically equivalent mappins and prune the mapping.",
    )
    max3d: PositiveFloat = Field(
        1000.0,
        description="Maximum discrepancy in Angstroms between atoms before the mapping is not allowed. Large numbers trim no atoms.",
    )
    element_change: bool = Field(
        True, description="Whether to allow element changes in the mappings."
    )
    seed: str = Field(
        "", description="The SMARTS string used as a seed for MCS searches."
    )
    shift: bool = Field(
        True,
        description="When determining 3D overlap, if to translate the two MCS to minimise the RMSD to boost potential alignment.",
    )

    def _get_mapper(self):
        from openfe import LomapAtomMapper

        return LomapAtomMapper(**self.dict(exclude={"type"}))

    def provenance(self) -> dict[str, str]:
        import lomap
        import openfe
        import rdkit

        return {
            "openfe": openfe.__version__,
            "lomap": lomap.__version__,
            "rdkit": rdkit.__version__,
        }


class PersesAtomMapper(_BaseAtomMapper):
    """
    A settings class for the PersesAtomMapper in openFE
    """

    type: Literal["PersesAtomMapper"] = "PersesAtomMapper"

    allow_ring_breaking: bool = Field(
        True, description="If only full cycles of the molecules should be mapped."
    )
    preserve_chirality: bool = Field(
        True,
        description="If mappings must strictly preserve the chirality of the molecules.",
    )
    use_positions: bool = Field(
        True,
        description="If 3D positions should be used during the generation of the mappings.",
    )
    coordinate_tolerance: PositiveFloat = Field(
        0.25,
        description="A tolerance on how close coordinates need to be in Angstroms before they can be mapped. Does nothing if use_positions is `False`.",
    )

    def _get_mapper(self):
        from openfe import PersesAtomMapper

        return PersesAtomMapper(**self.dict(exclude={"type"}))

    def provenance(self) -> dict[str, str]:
        import openeye.oechem
        import openfe
        import perses

        return {
            "openfe": openfe.__version__,
            "perses": perses.__version__,
            "openeye.oechem": openeye.oechem.OEChemGetVersion(),
        }


class KartographAtomMapper(_BaseAtomMapper):
    """
    A settings class for the kertograph atom mapping method.
    """

    type: Literal["KartographAtomMapper"] = "KartographAtomMapper"

    atom_ring_matches_ring: bool = Field(
        False, description="It only rings should be matched to other rings."
    )
    atom_max_distance: float = Field(
        0.95,
        description="The distance in Angstroms between two atoms before they can not be matched.",
    )
    atom_map_hydrogens: bool = Field(
        True, description="If hydrogens should also be mapped in the transform."
    )
    map_hydrogens_on_hydrogens_only: bool = Field(
        False, description="If hydrogens should only be matched to other hydrogens."
    )
    mapping_algorithm: Literal[
        "linear_sum_assignment", "minimal_spanning_tree"
    ] = Field(
        "linear_sum_assignment",
        description="The mapping algorithm that should be used.",
    )

    def _get_mapper(self):
        from kartograf.atom_mapper import KartografAtomMapper, mapping_algorithm

        # workaround the awkward argument name
        settings = self.dict(exclude={"type", "mapping_algorithm"})
        settings["_mapping_algorithm"] = (
            mapping_algorithm.linear_sum_assignment
            if self.mapping_algorithm == "linear_sum_assignment"
            else mapping_algorithm.minimal_spanning_tree
        )
        return KartografAtomMapper(**settings)

    def provenance(self) -> dict[str, str]:
        import kartograf._version
        import openfe
        import rdkit

        return {
            "openfe": openfe.__version__,
            "rdkit": rdkit.__version__,
            "kartograf": kartograf._version.__version__,
        }


class _NetworkPlannerSettings(_SchemaBase):
    """
    The Network planner settings which configure how the FEP networks should be constructed.
    """

    type: Literal["NetworkPlanner"] = "NetworkPlanner"

    atom_mapping_engine: Union[
        LomapAtomMapper, PersesAtomMapper, KartographAtomMapper
    ] = Field(
        LomapAtomMapper(),
        description="The method which should be used to create the mappings between molecules in the FEP network.",
    )
    scorer: Literal["default_lomap", "default_perses"] = Field(
        "default_lomap",
        description="The method which should be used to score the proposed atom mappings by the atom mapping engine.",
    )
    network_planning_method: Literal["radial", "maximal", "minimal_spanning"] = Field(
        "radial",
        description="The way in which the ligand network should be connected. Note radial requires a central ligand node.",
    )


class PlannedNetwork(_NetworkPlannerSettings):
    """
    A planned openFE network with complete atom mappings
    """

    type: Literal["PlannedNetwork"] = "PlannedNetwork"

    provenance: dict[str, str] = Field(
        ...,
        description="The provenance of the software used to generate ligand network.",
    )
    central_ligand: Optional[openfe.SmallMoleculeComponent] = Field(
        ..., description="The central ligand needed for radial networks."
    )
    ligands: list[openfe.SmallMoleculeComponent] = Field(
        ...,
        description="The list of docked ligands which should be included in the radial network.",
    )
    graphml: str = Field(
        ...,
        description="The GraphML string representation of the OpenFE LigandNetwork object. See to `to_ligand_network()`",
    )

    class Config:
        """Overwrite the class config to freeze the results model"""

        allow_mutation = False
        arbitrary_types_allowed = True

    def to_ligand_network(self) -> openfe.LigandNetwork:
        """
        Build a LigandNetwork from the planned network.
        """
        return openfe.LigandNetwork.from_graphml(self.graphml)


class NetworkPlanner(_NetworkPlannerSettings):
    """
    An FEP network factory which builds networks based on the configured settings.
    """

    type: Literal["NetworkPlanner"] = "NetworkPlanner"

    def _get_scorer(self):
        # We don't need to explicitly handle raising an error as pydantic validation will do it for us
        if self.scorer == "default_lomap":
            return openfe.lomap_scorers.default_lomap_score
        else:
            return openfe.perses_scorers.default_perses_scorer

    def _get_network_plan(self):
        if self.network_planning_method == "radial":
            return openfe.ligand_network_planning.generate_radial_network
        elif self.network_planning_method == "maximal":
            return openfe.ligand_network_planning.generate_maximal_network
        else:
            return openfe.ligand_network_planning.generate_minimal_spanning_network

    def generate_network(
        self,
        ligands: list[openfe.SmallMoleculeComponent],
        central_ligand: Optional[openfe.SmallMoleculeComponent] = None,
    ) -> PlannedNetwork:
        """
        Generate a network with the configured settings.

        Note:
            central_ligand is required for the radial type networks
        Args:
            ligands: The set of ligands which should be included in the network.
            central_ligand: The ligand which should be considered as the central node in a radial network
        """

        # validate the inputs
        if self.network_planning_method == "radial" and central_ligand is None:
            raise RuntimeError(
                "The radial type network requires a ligand to act as the central node."
            )

        # build the network planner
        planner_data = {
            "ligands": ligands,  # need to convert to rdkit objects?
            "mappers": [self.atom_mapping_engine.get_mapper()],
            "scorer": self._get_scorer(),
        }

        # add the central ligand if required
        if self.network_planning_method == "radial":
            planner_data["central_ligand"] = central_ligand

        network_method = self._get_network_plan()
        ligand_network = network_method(**planner_data)

        return PlannedNetwork(
            **self.dict(exclude={"type"}),
            ligands=ligands,
            central_ligand=central_ligand,
            graphml=ligand_network.to_graphml(),
            provenance=self.atom_mapping_engine.provenance(),
        )


class SolventSettings(_SchemaBase):
    """
    A settings class to encode the solvent used in the OpenFE FEP calculations.
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
    ion_concentration: PositiveFloat = Field(
        0.15, description="The ionic concentration required in molar units."
    )

    def to_solvent_component(self) -> gufe.SolventComponent:
        # work around units until we use openff-models
        solvent_data = self.dict(exclude={"type", "ion_concentration"})
        solvent_data["ion_concentration"] = self.ion_concentration * OFFUnit.molar
        return gufe.SolventComponent(**solvent_data)


class FreeEnergyPerturbationNetwork(_SchemaBase):
    """
    A schema of a FEP network created by the FreeEnergyPerturbationFactory which contains all runtime settings and can
    be converted to local openFE inputs or submitted to alchemiscale.
    """

    type: Literal["FreeEnergyPerturbationNetwork"] = "FreeEnergyPerturbationNetwork"
    dataset_name: str = Field(
        ...,
        description="The name of the dataset, this will be used for local files and the alchemiscale network.",
    )
    network: PlannedNetwork = Field(
        ...,
        description="The planned free energy network with atom mappings between ligands.",
    )
    receptor: openfe.ProteinComponent = Field(
        ..., description="The receptor which should be used in the FEP calculation."
    )
    solvent_settings: SolventSettings = Field(
        ...,
        description="The solvent settings which should be used during the free energy calculations.",
    )
    protocol: RelativeHybridTopologyProtocolSettings = Field(
        ..., description="The protocol with all runtime settings configured"
    )
    # TODO add more alchemiscale specific options to make it easy to find on the network.

    class Config:
        """Overwrite the class config to freeze the results model"""

        allow_mutation = False


class FreeEnergyPerturbationFactory(_SchemaBase):
    """A factory class to configure FEP calculations using the openFE pipeline. This generates a prepared FEP network
    which can be executed locally or submitted to Alchemiscale."""

    type: Literal["FreeEnergyPerturbationFactory"] = "FreeEnergyPerturbationFactory"

    network_planner: NetworkPlanner = Field(
        NetworkPlanner(),
        description="The network planner settings which should be used to construct the network.",
    )
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

    def _get_protocol(self):
        if self.protocol == "RelativeHybridTopologyProtocol":
            return RelativeHybridTopologyProtocolSettings

    def create_fep_dataset(
        self,
        dataset_name: str,
        receptor: openfe.ProteinComponent,
        ligands: list[openfe.SmallMoleculeComponent],
        central_ligand: Optional[openfe.SmallMoleculeComponent],
    ) -> FreeEnergyPerturbationNetwork:
        """
         Use the factory settings to create a FEP dataset using OpenFE models.

        Args:
            dataset_name: The name which should be given to this dataset, this will be used for local file creation or
            to identify on alchemiscale
            receptor: The prepared receptor to use in the FEP dataset.
            ligands: The list of prepared and state enumerated ligands to use in the FEP calculation.
            central_ligand: An optional ligand which should be considered as the center only needed for radial networks.
            Note this ligand will be deduplicated from the list if it appears in both.

         Returns:
             The planned FEP network which can be executed locally or submitted to alchemiscale.
        """

        # start by trying to plan the network
        planned_network = self.network_planner.generate_network(
            ligands=ligands, central_ligand=central_ligand
        )
        # transport all other settings to the network
        protocol = self._get_protocol()
        protocol_settings = protocol(
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
        planned_fep_network = FreeEnergyPerturbationNetwork(
            dataset_name=dataset_name,
            network=planned_network,
            receptor=receptor,
            solvent_settings=self.solvent_settings,
            protocol=protocol_settings,
        )
        return planned_fep_network
