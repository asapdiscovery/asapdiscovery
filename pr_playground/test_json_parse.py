from asapdiscovery.data.fitness import parse_fitness_json
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from pathlib import Path
import pandas as pd

HTMLVisualizer(["tmp_inputs/mac1_lig.sdf"],
               [Path("out_test.html")],
               "SARS-CoV-2-Mac1",
               Path("tmp_inputs/mac1_prot.pdb",),
               "subpockets",
               parse_fitness_json("SARS-CoV-2-Mac1")).write_pose_visualizations()
