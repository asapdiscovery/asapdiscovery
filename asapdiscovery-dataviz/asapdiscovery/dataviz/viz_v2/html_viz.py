from asapdiscovery.data.fitness import target_has_fitness_data
from pydantic import Field, root_validator
from asapdiscovery.dataviz.viz_v2.visualizer import VisualizerBase
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.docking.docking_v2 import DockingResult
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from asapdiscovery.data.openeye import save_openeye_pdb


from pathlib import Path
from enum import Enum
from tempfile import NamedTemporaryFile

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
    output_dir: Path = Field(
        "html", description="Output directory to write HTML files to"
    )

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

    def _visualize(self, docking_results: list[DockingResult]):
        """
        Visualize a list of docking results.

        NOTE: This is an extremely bad way of doing this, but it's a quick fix for now
        """

        # write out the docked pose
        for result in docking_results:
            output_pref = result.input_pair.complex.target.target_name + "_+_" + result.posed_ligand.compound_name + ".html"
            outpath = self.output_dir / output_pref
            with NamedTemporaryFile("w", suffix=".pdb") as 
            save_openeye_pdb(result.to_protein(), pdb_temp_path)
            pose_temp_path = NamedTemporaryFile(suffix=".sdf")
            result.posed_ligand.to_sdf(pose_temp_path)
            viz_class = HTMLVisualizer([pose_temp_path], [outpath], pdb_temp_path, self.colour_method, logger=None, debug=self.debug)
            outpaths = viz_class.write_pose_visualizations()
        # flatten 
        outpaths = [item for sublist in outpaths for item in sublist]
        return outpaths
        
    def provenance(self):
        return {}
