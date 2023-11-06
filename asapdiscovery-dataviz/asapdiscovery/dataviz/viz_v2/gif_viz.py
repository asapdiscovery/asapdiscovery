from asapdiscovery.data.fitness import target_has_fitness_data
from pydantic import Field, root_validator
from asapdiscovery.dataviz.viz_v2.visualizer import VisualizerBase
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.docking.docking_v2 import DockingResult
from enum import Enum


class GIFVisualizerV2(VisualizerBase):
    """
    Class for generating GIF visualizations of MD simulations.
    """

    ...
