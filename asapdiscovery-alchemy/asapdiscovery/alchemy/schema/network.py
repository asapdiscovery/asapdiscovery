import abc
from typing import Callable, Literal, Optional, Union

import openfe
from asapdiscovery.data.schema.ligand import Ligand
from openfe.setup import LigandNetwork
from pydantic.v1 import Field

from ._util import check_ligand_series_uniqueness_and_names
from .atom_mapping import KartografAtomMapper, LomapAtomMapper, PersesAtomMapper
from .base import _SchemaBase


class _NetworkPlannerMethod(_SchemaBase, abc.ABC):
    """
    The network planner method and settings which control the type of network produced.
    """

    type: Literal["_NetworkPlannerMethod"] = "_NetworkPlannerMethod"

    @abc.abstractmethod
    def get_planning_function(self) -> Callable:
        """
        Get the configured network planning function as functools partial which is ready to be called.
        Returns
        -------
            A callable network planning function with runtime options already configured.
        """
        ...


class RadialPlanner(_NetworkPlannerMethod):
    """Plan radial type networks using openfe."""

    type: Literal["RadialPlanner"] = "RadialPlanner"

    def get_planning_function(self) -> Callable:
        return openfe.ligand_network_planning.generate_radial_network


class MaximalPlanner(_NetworkPlannerMethod):
    """Plan maximally connected networks using openfe."""

    type: Literal["MaximalPlanner"] = "MaximalPlanner"

    def get_planning_function(self) -> Callable:
        return openfe.ligand_network_planning.generate_maximal_network


class MinimalSpanningPlanner(_NetworkPlannerMethod):
    """Plan minimal spanning networks where each ligand has a single connection to the graph with openfe."""

    type: Literal["MinimalSpanningPlanner"] = "MinimalSpanningPlanner"

    def get_planning_function(self) -> Callable:
        return openfe.ligand_network_planning.generate_minimal_spanning_network


class MinimalRedundantPlanner(_NetworkPlannerMethod):
    """Planing minimally spanning networks with configurable redundancy using openfe."""

    type: Literal["MinimalRedundantPlanner"] = "MinimalRedundantPlanner"

    redundancy: int = Field(
        2,
        description="The number of minimal spanning networks which should be combined to make redundant networks.",
        gt=1,
    )

    def get_planning_function(self) -> Callable:
        import functools

        return functools.partial(
            openfe.ligand_network_planning.generate_minimal_redundant_network,
            mst_num=self.redundancy,
        )


class CustomNetworkPlanner(_NetworkPlannerMethod):
    """Plan a user-set custom network"""

    type: Literal["CustomNetworkPlanner"] = "CustomNetworkPlanner"

    edges: list[tuple[str, str]] = Field(
        ...,
        description="A list of tuples with ligand names which define the transformation edges.",
    )

    def get_planning_function(self) -> Callable:
        def _plan_from_names(ligands, mappers, *args, **kwargs):
            # format the data to fit the planing method
            data = {"ligands": ligands, "mapper": mappers[0], "names": self.edges}
            return openfe.ligand_network_planning.generate_network_from_names(**data)

        return _plan_from_names


class _NetworkPlannerSettings(_SchemaBase):
    """
    The Network planner settings which configure how the FEC networks should be constructed.
    """

    type: Literal["NetworkPlanner"] = "NetworkPlanner"

    atom_mapping_engine: Union[
        LomapAtomMapper, PersesAtomMapper, KartografAtomMapper
    ] = Field(
        LomapAtomMapper(),
        description="The method which should be used to create the mappings between molecules in the FEC network.",
    )
    scorer: Literal["default_lomap", "default_perses"] = Field(
        "default_lomap",
        description="The method which should be used to score the proposed atom mappings by the atom mapping engine.",
    )
    network_planning_method: Union[
        RadialPlanner,
        MaximalPlanner,
        MinimalSpanningPlanner,
        MinimalRedundantPlanner,
        CustomNetworkPlanner,
    ] = Field(
        MinimalRedundantPlanner(),
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
    central_ligand: Optional[Ligand] = Field(
        ..., description="The central ligand needed for radial networks."
    )
    ligands: list[Ligand] = Field(
        ...,
        description="The list of docked ligands which should be included in the network.",
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

    def to_openfe_ligands(self) -> list[openfe.SmallMoleculeComponent]:
        """
        Convert all ligand sdfs back to the openfe SmallMoleculeComponent including the central ligand if present.

        Returns:
            A list of openfe.SmallMoleculeComponents made from the central ligand followed by all ligands in the network.
        """
        ligands = [mol.to_openfe() for mol in self.ligands]
        if self.central_ligand is not None:
            ligands.insert(0, self.central_ligand.to_openfe())
        return ligands

    @classmethod
    def from_graphml(cls, graphml: str) -> "PlannedNetwork":
        """
        Build a PlannedNetwork from a graphml string.

        Args:
            graphml: The graphml string representation of the network.

        Returns:
            A PlannedNetwork object with the ligands and central ligand if present.
        """
        ligand_network = LigandNetwork.from_graphml(graphml)
        if not ligand_network.is_connected():
            raise ValueError(
                "The provided graphml does not represent a connected network."
            )
        # extract ligands from the network
        small_molecule_components = ligand_network.nodes

        ligands = [Ligand.from_openfe(mol) for mol in small_molecule_components]

        check_ligand_series_uniqueness_and_names(ligands)

        provenance = {"source": "pre-generated", "graphml": graphml}
        return cls(
            ligands=ligands,
            central_ligand=None,
            graphml=graphml,
            provenance=provenance,
        )


class NetworkPlanner(_NetworkPlannerSettings):
    """
    An FEC network factory which builds networks based on the configured settings.
    """

    type: Literal["NetworkPlanner"] = "NetworkPlanner"

    def _get_scorer(self):
        # We don't need to explicitly handle raising an error as pydantic validation will do it for us
        if self.scorer == "default_lomap":
            return openfe.lomap_scorers.default_lomap_score
        else:
            return openfe.perses_scorers.default_perses_scorer

    def generate_network(
        self,
        ligands: list[Ligand],
        central_ligand: Optional[Ligand] = None,
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
        if (
            self.network_planning_method.type == "RadialPlanner"
            and central_ligand is None
        ):
            raise RuntimeError(
                "The radial type network requires a ligand to act as the central node."
            )

        # build the network planner
        planner_data = {
            "ligands": [
                mol.to_openfe() for mol in ligands
            ],  # need to convert to rdkit objects?
            "mappers": [self.atom_mapping_engine.get_mapper()],
            "scorer": self._get_scorer(),
        }

        # add the central ligand if required
        if self.network_planning_method.type == "RadialPlanner":
            planner_data["central_ligand"] = central_ligand.to_openfe()

        network_method = self.network_planning_method.get_planning_function()
        ligand_network = network_method(**planner_data)

        return PlannedNetwork(
            **self.dict(exclude={"type"}),
            ligands=ligands,
            central_ligand=central_ligand,
            graphml=ligand_network.to_graphml(),
            provenance=self.atom_mapping_engine.provenance(),
        )
