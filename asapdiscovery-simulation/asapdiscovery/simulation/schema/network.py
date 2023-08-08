from typing import Literal, Optional, Union

import openfe
from asapdiscovery.simulation.schema.atom_mapping import (
    KartografAtomMapper,
    LomapAtomMapper,
    PersesAtomMapper,
)
from asapdiscovery.simulation.schema.base import _SchemaBase
from pydantic import Field


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
    network_planning_method: Literal["radial", "maximal", "minimal_spanning"] = Field(
        "minimal_spanning",
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
    central_ligand: Optional[str] = Field(
        ..., description="The central ligand needed for radial networks."
    )
    ligands: list[str] = Field(
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

    def to_openfe_ligands(self) -> list[openfe.SmallMoleculeComponent]:
        """
        Convert all ligand sdfs back to the openfe SmallMoleculeComponent including the central ligand if present.

        Returns:
            A list of openfe.SmallMoleculeComponents made from the central ligand followed by all ligands in the network.
        """
        ligands = [
            openfe.SmallMoleculeComponent.from_sdf_string(ligand_str)
            for ligand_str in self.ligands
        ]
        if self.central_ligand is not None:
            ligands.insert(
                0, openfe.SmallMoleculeComponent.from_sdf_string(self.central_ligand)
            )
        return ligands


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
            ligands=[ligand.to_sdf() for ligand in ligands],
            central_ligand=central_ligand
            if central_ligand is None
            else central_ligand.to_sdf(),
            graphml=ligand_network.to_graphml(),
            provenance=self.atom_mapping_engine.provenance(),
        )
