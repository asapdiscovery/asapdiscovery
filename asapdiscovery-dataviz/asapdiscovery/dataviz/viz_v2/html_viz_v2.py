
from asapdiscovery.data.fitness import parse_fitness_json, target_has_fitness_data
from pydantic import BaseModel, Field, root_validator
from asapdiscovery.dataviz.visualize_v2.visualize_v2VisualizerBase



class ColourMethod(str, Enum):
    subpockets = "subpockets"
    fitness = "fitness"


class HTMLVisualizerV2(BaseModel):
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





