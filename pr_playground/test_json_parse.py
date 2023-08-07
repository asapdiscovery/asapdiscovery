from asapdiscovery.data.fitness import parse_fitness_json
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from pathlib import Path
import pandas as pd

HTMLVisualizer(["tmp_inputs/Mpro_combined.sdf"], 
               [Path("out_test.html")], 
               "SARS-CoV-2-Mpro",
               Path("tmp_inputs/p0045_prot.pdb",),
               "bfactor",
               parse_fitness_json("SARS-CoV-2-Mpro")).write_pose_visualizations()