import logging
from enum import Enum
from pathlib import Path

from asapdiscovery.data.fitness import target_has_fitness_data
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from asapdiscovery.dataviz.viz_v2.visualizer import VisualizerBase
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.docking.docking_v2 import DockingResult
from pydantic import Field, root_validator

logger = logging.getLogger(__name__)


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

    def get_tag_for_colour_method(self):
        """
        Get the tag to use for the colour method.
        """
        if self.colour_method == ColourMethod.subpockets:
            return DockingResultCols.HTML_PATH_POSE.value
        elif self.colour_method == ColourMethod.fitness:
            return DockingResultCols.HTML_PATH_FITNESS.value

    def _visualize(self, docking_results: list[DockingResult]) -> list[dict[str, str]]:
        """
        Visualize a list of docking results.

        NOTE: This is an extremely bad way of doing this, but it's a quick fix for now
        """
        data = []
        for result in docking_results:
            # sorryyyyy
            output_pref = result.unique_name()
            outpath = self.output_dir / output_pref / "pose.html"
            viz_class = HTMLVisualizer(
                [result.posed_ligand.to_oemol()],
                [outpath],
                self.target,
                result.to_protein(),
                self.colour_method,
                debug=self.debug,
                align=False,
            )
            outpaths = viz_class.write_pose_visualizations()
            if len(outpaths) != 1:
                raise ValueError(
                    f"Expected 1 HTML file to be written, but got {len(outpaths)}"
                )
            # make dataframe with ligand name, target name, and path to HTML
            row = {}
            row[
                DockingResultCols.LIGAND_ID.value
            ] = result.input_pair.ligand.compound_name
            row[
                DockingResultCols.TARGET_ID.value
            ] = result.input_pair.complex.target.target_name
            row[DockingResultCols.SMILES.value] = result.input_pair.ligand.smiles
            row[self.get_tag_for_colour_method()] = outpaths[0]
            data.append(row)

        return data

    def provenance(self):
        return {}
