
from asapdiscovery.data.fitness import target_has_fitness_data
from pydantic import Field, root_validator
from asapdiscovery.dataviz.viz_v2.visualizer_v2 import VisualizerBase
from asapdiscovery.dataviz.html_viz import HTMLVisualizer



class ColourMethod(str, Enum):
    subpockets = "subpockets"
    fitness = "fitness"


class HTMLVisualizerV2(VisualizerBase):
    """
    Class for generating HTML visualizations of poses.
    Currently a translation layer between the old and new visualizers with the API we want moving forward.
    """

    target: TargetTags = Field(..., description="Target to visualize poses for")
    colour_method: ColourMethod = Field(
        ColourMethod.subpockets,
        description="Protein surface coloring method. Can be either by `subpockets` or `fitness`",
    )
    debug: bool = Field(False, description="Whether to run in debug mode")
    output_dir: Path = Field("poses", description="Directory to write poses to")

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

    def _visualize(self, docking_results: list[DockingResult]) -> pd.DataFrame:
        """
        Visualize a list of docking results.
        """
        ligs = [docking_result.input_pair.ligand for docking_result in docking_results]
        names_unique = compound_names_unique(ligs)
        # if names are not unique, we will use unknown_ligand_{i} as the ligand portion of directory
        # when writing files

        # write out the docked pose
        for i, result in enumerate(docking_results):
            if (
                not result.input_pair.ligand.compound_name
                == result.posed_ligand.compound_name
            ):
                raise ValueError(
                    "Compound names of input pair and posed ligand do not match"
                )
            if names_unique:
                output_pref = (
                    result.input_pair.complex.target.target_name
                    + "_+_"
                    + result.posed_ligand.compound_name
                )
            else:
                output_pref = (
                    result.input_pair.complex.target.target_name
                    + "_+_"
                    + f"unknown_ligand_{i}"
                )
            
            pose_temp = tempfile.NamedTemporaryFile(suffix=".pdb")
            







            
            viz_class = HTMLVisualizer([pose_path], outpath, self.target, protein, self.colour_method, logger=None, debug=self.debug)
            outpaths = viz_class.write_pose_visualizations()





