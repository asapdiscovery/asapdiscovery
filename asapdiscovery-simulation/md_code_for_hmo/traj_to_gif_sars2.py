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
# TODO: Refine --pse (rename) - this should be a bool. Currently the filepath is not being used.
__doc__ = """Creates a light-weight animated GIF file from a set of simulation output files.

Usage:
  traj_to_gif.py --system=FILE --traj=FILE --gif=FILE [--pse] [--smooth=INT] [--contacts] [--pse_share] [--interval=INT]
  traj_to_gif.py (-h | --help)

Options:
  -h --help           Show this screen.
  --system=FILE       System PDB filename.
  --traj=FILE         Trajectory DCD filename.
  --gif=FILE          Animated GIF filename.
  --pse               Write PyMol session (.pse) states to filename (for debugging)
  --pse_share         Write PyMol session (.pse) of ligand-protein system in colored perspective to filename (for results sharing)
  --smooth=INT        If specified, will smooth frames with the specified window
  --contacts          If specified, show contacts (requires show_contacts.py plugin)
  --interval=INT      If specified, load frames with specified interval.
"""
from docopt import docopt
arguments = docopt(__doc__, version='simulate 0.1')

#TODO: migrate below to e.g. yaml file:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#SARS2
view_coords = (\
    -0.393657833,   -0.737275898,   -0.549044371,\
     0.308091789,   -0.668542325,    0.676846981,\
    -0.866092026,    0.097289830,    0.490323097,\
    -0.000008677,    0.000024922,  -58.667091370,\
    12.463963509,   -6.047869682,   18.167898178,\
    13.353301048,  103.972755432,  -20.000000000 )
##MERS
#view_coords = (\
#    -0.635950804,   -0.283323288,   -0.717838645,\
#    -0.040723491,   -0.916550398,    0.397835642,\
#    -0.770651817,    0.282238036,    0.571343124,\
#     0.000061535,    0.000038342,  -58.079559326,\
#     8.052228928,    0.619271040,   21.864795685,\
#  -283.040344238,  399.190032959,  -20.000000000 )

##7ENE
#view_coords = (\
#     0.710110664,    0.317291290,   -0.628544748,\
#    -0.485409290,   -0.426031262,   -0.763462067,\
#    -0.510019004,    0.847245276,   -0.148514912,\
#     0.000047840,    0.000042381,  -58.146995544,\
#    -1.610392451,   -9.301478386,   57.779785156,\
#    13.662354469,  102.618484497,  -20.000000000 )

# set colorings of subpockets by resn. This may change over time.
pocket_dict = { # SARS2
"subP1" : "140-145+163+172",
"subP1_prime" : "25-27",
"subP2" : "41+49+54",
"subP3_4_5" : "165-168+189-192",
"sars_unique" : "25+49+142+164+168+169+181+186+188+190+191",
}

#pocket_dict = { # MERS
#"subP1" : "143+144+145+146+147+148+166+175",
#"subP1_prime" : "25+26+27",
#"subP2" : "41+49+54",
#"subP3_4_5" : "168+169+170+171+192+193+194+195",
#"sars_unique" : "25+49+145+167+171+172+184+189+191+193+194",
#}

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
complex_name = 'complex'
cmd.load(arguments['--system'], object=complex_name)

if arguments['--pse']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_1_loaded_system.pse...")
    cmd.save("session_1_loaded_system.pse")

# now select the residues, name them and color them.
for subpocket_name, residues in pocket_dict.items():
    cmd.select(subpocket_name,  f"{complex_name} and resi {residues} and polymer.protein")

for subpocket_name, color in color_dict.items():
    cmd.set("surface_color", color, f"({subpocket_name})")
if arguments['--pse']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_2_colored_subpockets.pse...")
    cmd.save("session_2_colored_subpockets.pse")

# Select ligand and receptor
cmd.select('ligand', 'resn UNK')
cmd.select('receptor', 'chain A or chain B') # TODO: Modify this to generalize to dimer
cmd.select('binding_site', 'name CA within 7 of resn UNK') # automate selection of the binding site

## set a bunch of stuff for visualization
cmd.set("bg_rgb", "white")
cmd.set("surface_color", "grey90")
cmd.bg_color("white")
cmd.hide("everything")
cmd.show("cartoon")
cmd.show("surface", 'receptor')
cmd.set('surface_mode', 3)
cmd.set("cartoon_color", "grey")
cmd.set("transparency", 0.3)
cmd.hide("surface", 'ligand') # for some reason sometimes a ligand surface is applied - hide this.


## select the ligand and subpocket residues, show them as sticks w/o nonpolar Hs
cmd.select("resn UNK")
cmd.show("sticks", "sele")
cmd.show("sticks", "subP*")
cmd.hide("sticks", "(elem C extend 1) and (elem H)")
cmd.color("pink", "elem C and sele")

cmd.set_view(view_coords)
if arguments['--pse'] or arguments['--pse_share']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_3_set_ligand_view.pse...")
    cmd.save("session_3_set_ligand_view.pse")

## load trajectory; center the system in the simulation and smoothen between frames.
log.info(':thinking_face:  Loading trajectory into PyMol...')
interval = 1
if arguments['--interval']:
    interval = arguments['--interval']
cmd.load_traj(f"{arguments['--traj']}", object=complex_name, start=1, interval=interval)
if arguments['--pse']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_4_loaded_trajectory.pse...")
    cmd.save("session_4_loaded_trajectory.pse")

log.info(':triangular_ruler:  Intrafitting simulation...')
cmd.intra_fit("binding_site")
if arguments['--smooth']:
    cmd.smooth('all', window=int(arguments['--smooth'])) # perform some smoothing of frames
cmd.zoom('resn UNK', buffer=1) # zoom to ligand

if arguments['--contacts']:
    from show_contacts import show_contacts
    #cmd.run('show_contacts.py')
    show_contacts('ligand', 'receptor')

#cmd.set("ray_shadows", 0) 
if arguments['--pse']:
    log.info(f":page_facing_up:  Writing PyMol ensemble to session_5_intrafitted.pse...")
    cmd.save("session_5_intrafitted.pse")

# Process the trajectory in a temporary directory
import tempfile
from pygifsicle import optimize
with tempfile.TemporaryDirectory() as tmpdirname:
    log.info(f':file_folder: Creating temporary directory {tmpdirname}')

    ## now make the movie. 
    log.info(':camera_with_flash:  Rendering images for frames...')
    cmd.set("ray_trace_frames", 0) # ray tracing with surface representation is too expensive.
    cmd.set("defer_builds_mode", 1) # this saves memory for large trajectories
    cmd.set("cache_frames", 0)
    cmd.set("max_threads", 4) # limit to 4 threads to prevent PyMOL from oversubscribing
    cmd.mclear()   # clears cache 
    prefix = f"{tmpdirname}/frame"
    cmd.mpng(prefix)   # saves png of each frame as "frame001.png, frame002.png, .."
    # TODO: higher resolution on the pngs.
    # TODO: Find way to improve writing speed by e.g. removing atoms not in view. Currently takes ~80sec per .png

    ## use imagio to create a gif from the .png files generated by pymol
    import imageio.v2 as iio
    from glob import glob
    gif_filename = arguments["--gif"]
    log.info(f':videocassette:  Creating animated GIF {gif_filename} from images...')
    png_files = glob(f"{prefix}*.png")
    
    if len(png_files) == 0:
        raise IOError(f"No {prefix}*.png files found - did PyMol not generate any?")
    
    png_files.sort() # for some reason *sometimes* this list is scrambled messing up the GIF. Sorting fixes the issue.

    png_files = png_files[-100:] # take only last .5ns of trajectory to get nicely equilibrated pose.
    with iio.get_writer(gif_filename, mode='I') as writer:
        for filename in png_files:
            image = iio.imread(filename)
            writer.append_data(image)
    
    # now compress the GIF with the method that imagio recommends (https://imageio.readthedocs.io/en/stable/examples.html).
    log.info(':gift:  Compressing animated gif...')
    optimize(gif_filename) # this is in-place.
