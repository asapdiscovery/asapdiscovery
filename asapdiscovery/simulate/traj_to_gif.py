"""
Creates a light-weight animated GIF file from a simulation.
NB: runs on CPU
"""

# Configure logging
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
from rich.console import Console
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
log = logging.getLogger("rich")

# Use docopt for CLI handling
# TODO: Once we refactor this to encapsulate behavior in functions (or classes) migrate to click: https://click.palletsprojects.com/en/8.1.x/
# TODO: Refine --writesess (rename) - this should be a bool. Currently the filepath is not being used.
__doc__ = """Creates a light-weight animated GIF file from a set of simulation output files.

Usage:
  traj_to_gif.py --system=FILE --traj=FILE [--writesess=FILE]  
  traj_to_gif.py (-h | --help)

Options:
  -h --help           Show this screen.
  --system=FILE       System PDB filename.
  --traj=FILE         Trajectory DCD filename.
  --writesess=FILE    Write PyMol session states to filename (for debugging)

"""
from docopt import docopt
arguments = docopt(__doc__, version='simulate 0.1')

#TODO: migrate below to e.g. yaml file:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

view_coords = (\
     0.383985907,    0.676508129,    0.628391743,\
     0.267365783,   -0.732870340,    0.625607014,\
     0.883769155,   -0.072211117,   -0.462298840,\
     0.000457883,   -0.000074564,  -94.866157532,\
    -6.947262764,   -1.080027819,  -22.212558746,\
  -421.704925537,  611.407409668,  -20.000000000 )
## set colorings of subpockets by resn. This may change over time.
pocket_dict = { # SARS2
"subP1" : "140-145+163+172",
"subP1_prime" : "25-27",
"subP2" : "41+49+54",
"subP3_4_5" : "165-168+189-192",
"sars_unique" : "25+49+142+164+168+169+181+186+188+190+191",
}

# pocket_dict = { # MERS
# "subP1" : "143+144+145+146+147+148+166+175",
# "subP1_prime" : "25+26+27",
# "subP2" : "41+49+54",
# "subP3_4_5" : "168+169+170+171+192+193+194+195",
# "sars_unique" : "25+49+145+167+171+172+184+189+191+193+194",
# }

color_dict = {
"subP1" : "yellow",
"subP1_prime" : "orange",
"subP2" : "skyblue",
"subP3_4_5" : "aquamarine" 
} 
# TODO: pick color-blind-friendly scheme, e.g. using https://colorbrewer2.org/#type=qualitative&scheme=Pastel1&n=4
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## load pdb file of interest. Currently needs minimized PDB. 
from pymol import cmd
log.info(':thinking_face:  Loading system into PyMol and applying aesthetics...')
cmd.load(arguments['--system'])

if arguments['--writesess']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_1_loaded_system.pse...")
    cmd.save("session_1_loaded_system.pse")

# now select the residues, name them and color them.
for subpocket_name, residues in pocket_dict.items():
    cmd.select(subpocket_name,  f"{arguments['--system'].replace('.pdb','')} and resi {residues} and polymer.protein")

for subpocket_name, color in color_dict.items():
    cmd.set("surface_color", color, f"({subpocket_name})")
if arguments['--writesess']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_2_colored_subpockets.pse...")
    cmd.save("session_2_colored_subpockets.pse")

## set a bunch of stuff for visualization
cmd.set("bg_rgb", "white")
cmd.set("surface_color", "grey90")
cmd.bg_color("white")
cmd.hide("everything")
cmd.show("cartoon")
cmd.show("surface")
cmd.set("cartoon_color", "grey")
cmd.set("transparency", 0.3)

## select the ligand and subpocket residues, show them as sticks w/o nonpolar Hs
cmd.select("resn UNK")
cmd.show("sticks", "sele")
cmd.show("sticks", "subP*")
cmd.hide("sticks", "(elem C extend 1) and (elem H)")
cmd.color("pink", "elem C and sele")

cmd.set_view(view_coords)
if arguments['--writesess']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_3_set_ligand_view.pse...")
    cmd.save("session_3_set_ligand_view.pse")

## load trajectory; center the system in the simulation and smoothen between frames.
log.info(':thinking_face:  Loading trajectory into PyMol...')
cmd.load(f"{arguments['--traj']}")
if arguments['--writesess']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_4_loaded_trajectory.pse...")
    cmd.save("session_4_loaded_trajectory.pse")

log.info(':triangular_ruler:  Intrafitting simulation...')
cmd.intra_fit("resn UNK")
if arguments['--writesess']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_5_intrafitted.pse...")
    cmd.save("session_5_intrafitted.pse")

## now make the movie. 
log.info(':camera_with_flash:  Rendering images for frames...')
cmd.set("ray_trace_frames", 0) # ray tracing with surface representation is too expensive.
cmd.set("cache_frames", 0)
cmd.mclear()   # clears cache 
cmd.mpng("mov")   # saves png of each frame as "mov001.png, mov002.png, .."
# TODO: higher resolution on the pngs.
# TODO: Find way to improve writing speed by e.g. removing atoms not in view. Currently takes ~80sec per .png

## use imagio to create a gif from the .png files generated by pymol
import imageio.v2 as iio
from glob import glob
log.info(':videocassette:  Creating animated GIF from images...')
png_files = glob("tmp_mov*.png")
if len(png_files) == 0:
    raise IOError("No tmp_mov*.png files found - did PyMol not generate any?")
    
with iio.get_writer('output_simulation.gif', mode='I') as writer:
    for filename in png_files:
        image = iio.imread(filename)
        writer.append_data(image)

## remove all .png files. Could consider setting argument to not remove these during debugging?
import os
[ os.remove(f) for f in glob("tmp_mov*.png") ]