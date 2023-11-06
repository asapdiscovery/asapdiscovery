from asapdiscovery.data.fitness import target_has_fitness_data
from pydantic import Field, root_validator
from asapdiscovery.dataviz.viz_v2.visualizer import VisualizerBase
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.docking.docking_v2 import DockingResult
from enum import Enum


class ColourMethod(str, Enum):
    subpockets = "subpockets"
    fitness = "fitness"


class HTMLVisualizerV2(VisualizerBase):
    """
    Class for generating HTML visualizations of poses.
    """

    target: TargetTags = Field(..., description="Target to visualize poses for")
    colour_method: ColourMethod = Field(
        ColourMethod.subpockets,
        description="Protein surface coloring method. Can be either by `subpockets` or `fitness`",
    )
    debug: bool = Field(False, description="Whether to run in debug mode")

    @root_validator
    @classmethod
    def must_have_fitness_data(cls, values):
        target = values.get("target")
        colour_method = values.get("colour_method")
        if colour_method == ColourMethod.fitness and not target_has_fitness_data(
            target
        ):
            raise ValueError(
                f"Attempting to colour by fitness and {target} does not have fitness data, use `subpockets` instead."
            )
        return values

    def _visualize(self, inputs: list[DockingResult]):
        """
        Visualize a list of docking results.
        """

    def provenance(self):
        return {}
