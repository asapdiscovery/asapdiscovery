import logging
import shutil
from pathlib import Path
from typing import Union  # noqa: F401

from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.postera.manifold_data_validation import (
    TargetProteinMap,
    TargetTags,
)

from ._gif_blocks import GIFBlockData
from .resources.fonts import opensans_regular
from .show_contacts import show_contacts

logger = logging.getLogger(__name__)


class GIFVisualizer:
    """
    Class for generating GIF visualizations of MD trajectories.
    """

    allowed_targets = TargetTags.get_values()

    # TODO: replace input with a schema rather than paths.
    def __init__(
        self,
        trajectories: list[Path],
        systems: list[Path],
        output_paths: list[Path],
        target: str,
        frames_per_ns: int = 200,
        pse: bool = False,
        pse_share: bool = False,  # set to True for GIF viz debugging
        smooth: int = 3,
        contacts: bool = True,
        static_view_only: bool = False,
        start: int = 1,
        stop: int = -1,
        interval: int = 1,
        debug: bool = False,
    ):
        """
        Parameters
        ----------
        trajectories : List[Path]
            List of trajectories to visualize.
        systems : List[Path]
            List of matching PDB files to load the system from
        output_paths : List[Path]
            List of paths to write the visualizations to.
        target : str
            Target to visualize poses for. Must be one of the allowed targets
        pse : bool
            Whether to write PyMol session files.
        smooth : int
            Number of frames to smooth over.
        contacts : bool
            Whether to show contacts.
        static_view_only : bool
            If True, only write out a .PSE file with the canonical view; don't load trajectory
        start : int
            Start frame to load
        stop : int
            Stop frame to load
        interval : int
            Interval between frames to load
        debug : bool
            Whether to run in debug mode.

        """
        if not len(trajectories) == len(output_paths):
            raise ValueError("Number of trajectories and paths must be equal.")

        if target not in self.allowed_targets:
            raise ValueError(f"Target must be one of: {self.allowed_targets}")
        self.target = target
        logger.info(f"Visualizing trajectories for {self.target}")

        # setup view_coords, pocket_dict and color_dict for target

        self.view_coords = GIFBlockData.get_view_coords()

        self.pocket_dict = GIFBlockData.get_pocket_dict(self.target)

        self.color_dict = GIFBlockData.get_color_dict(self.target)

        self.trajectories = []
        self.output_paths = []
        self.systems = []
        for trajectory, system, path in zip(trajectories, systems, output_paths):
            if (
                trajectory
                and Path(trajectory).exists()
                and system
                and Path(system).exists()
            ):
                self.trajectories.append(trajectory)
                self.systems.append(system)
                self.output_paths.append(path)
            elif system and Path(system).exists():
                self.systems.append(system)
                self.output_paths.append(path)
                logger.critical("No trajectory provided - skipping GIF viz.")
            else:
                logger.critical(
                    f"Trajectory {trajectory} or system {system} does not exist - skipping."
                )

        # kwargs
        self.frames_per_ns = frames_per_ns
        self.pse = pse
        self.pse_share = pse_share
        self.smooth = smooth
        self.contacts = contacts
        self.start = start
        self.stop = stop
        self.interval = interval
        self.static_view_only = static_view_only
        self.debug = debug

        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Running in debug mode, setting pse=True")
            self.pse = True
        logger.debug(f"Writing GIF visualizations for {len(self.output_paths)} ligands")

    def write_traj_visualizations(self):
        """
        Write GIF visualizations for all trajectories.
        """
        output_paths = []
        for traj, system, path in zip(
            self.trajectories, self.systems, self.output_paths
        ):
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            output_path = self.write_traj_visualization(traj, system, path)
            output_paths.append(output_path)
        return output_paths

    def write_traj_visualization(self, traj, system, path):
        """
        Write GIF visualization for a single trajectory.
        """
        # NOTE very important, need to spawn a new pymol proc for each trajectory
        # when working in parallel, otherwise they will trip over each other and not work.

        import pymol2

        p = pymol2.PyMOL()
        p.start()

        parent_path = (
            path.parent
        )  # use parent so we can write the pse files to the same directory

        tmpdir = parent_path / "tmp"
        logger.info(f"Creating temporary directory {tmpdir}")
        tmpdir.mkdir(parents=True, exist_ok=True)

        complex_name = "complex"
        p.cmd.load(str(system), object=complex_name)

        if self.static_view_only:
            # this may be unprepped/unaligned, so need to align to master structure before writing out.
            reference_structure = master_structures[self.target]
            p.cmd.load(reference_structure, object="reference_master")
            p.cmd.align(complex_name, "reference_master")
            p.cmd.delete("reference_master")

        if self.pse:
            logger.info("Writing PyMol ensemble to session_1_loaded_system.pse...")
            p.cmd.save(str(parent_path / "session_1_loaded_system.pse"))

        # now select the residues, name them and color them.
        for subpocket_name, residues in self.pocket_dict.items():
            p.cmd.select(
                subpocket_name,
                f"{complex_name} and resi {residues} and polymer.protein",
            )

        for subpocket_name, color in self.color_dict.items():
            p.cmd.set("surface_color", color, f"({subpocket_name})")

        if self.pse:
            logger.info("Writing PyMol ensemble to session_2_colored_subpockets.pse")
            p.cmd.save(str(parent_path / "session_2_colored_subpockets.pse"))

        # Select ligand and receptor
        p.cmd.select("ligand", "resn UNK or resn LIG")
        p.cmd.select(
            "receptor", "chain A or chain B or chain 1 or chain 2"
        )  # TODO: Modify this to generalize to dimer
        p.cmd.select(
            "binding_site", "name CA within 7 of resn UNK or name CA within 7 resn LIG"
        )  # automate selection of the binding site

        # set a bunch of stuff for visualization
        p.cmd.set("bg_rgb", "white")
        p.cmd.set("surface_color", "grey90")
        p.cmd.bg_color("white")
        p.cmd.hide("everything")
        p.cmd.show("cartoon")
        p.cmd.show("surface", "receptor")
        p.cmd.set("surface_mode", 3)
        p.cmd.set("cartoon_color", "grey")
        p.cmd.set("transparency", 0.3)
        p.cmd.hide(
            "surface", "ligand"
        )  # for some reason sometimes a ligand surface is applied - hide this.

        # select the ligand and subpocket residues, show them as sticks w/o nonpolar Hs
        p.cmd.select("resn UNK or resn LIG")
        p.cmd.show("sticks", "sele")
        p.cmd.show("sticks", "subP*")
        p.cmd.hide("sticks", "(elem C extend 1) and (elem H)")
        p.cmd.color("pink", "elem C and sele")

        for subpocket_name, color in self.color_dict.items():
            # set non-polar sticks for this subpocket, color the backbone by subpocket color.
            p.cmd.select(subpocket_name)
            p.cmd.show("sticks", "sele")
            p.cmd.set("stick_color", color, f"({subpocket_name})")
            p.cmd.hide("sticks", "(elem C extend 1) and (elem H)")

        if self.pse:
            p.cmd.save(str(parent_path / "session_3_set_ligand_view.pse"))

        if not self.static_view_only:
            # load trajectory; center the system in the simulation and smoothen between frames.
            p.cmd.load_traj(
                str(traj),
                object=complex_name,
                start=self.start,
                stop=self.stop,
                interval=self.interval,
            )
            if self.pse:
                p.cmd.save(str(parent_path / "session_4_loaded_trajectory.pse"))

            # center the system to the minimized structure
            # reload
            complex_name_min = "complex_min"
            p.cmd.load(str(system), object=complex_name_min)

            logger.info("Aligning simulation...")
            if self.smooth:
                p.cmd.smooth(
                    "all", window=int(self.smooth)
                )  # perform some smoothing of frames

        if self.contacts:
            logger.info("Showing contacts...")
            p.cmd.extract("ligand_obj", "ligand")

            show_contacts(p, "ligand_obj", "receptor")

        p.cmd.set_view(self.view_coords)  # sets general orientation
        if self.static_view_only:
            p.cmd.save(str(path))
            # remove tmpdir
            shutil.rmtree(tmpdir)
            return path  # for static view we can end the function here.

        # turn on depth cueing
        p.cmd.set("depth_cue", 1)

        # now, select stuff to hide; we select everything that is
        # farther than 15 Ang from our ligand.
        p.cmd.select("th", "(all) and not ( (all) within 15 of ligand_obj)")
        # hide it to save rendering time.
        p.cmd.hide("everything", "th")

        # capsid needs clipping planes as the ligand is encapsulated
        if TargetProteinMap[self.target] == "Capsid":
            p.cmd.clip("near", -25)

        if self.pse or self.pse_share:
            p.cmd.save(str(parent_path / "session_5_selections.pse"))

        p.cmd.align(complex_name, complex_name_min)
        p.cmd.delete(complex_name_min)
        # Process the trajectory in a temporary directory
        from pygifsicle import gifsicle

        # now make the movie.
        logger.info("Rendering images for frames...")
        p.cmd.set(
            "ray_trace_frames", 0
        )  # ray tracing with surface representation is too expensive.
        p.cmd.set("defer_builds_mode", 1)  # this saves memory for large trajectories
        p.cmd.set("cache_frames", 0)
        p.cmd.set(
            "max_threads", 1
        )  # limit to 1 threads to prevent PyMOL from oversubscribing
        p.cmd.mclear()  # clears cache
        prefix = str(tmpdir / "frame")
        p.cmd.mpng(
            prefix
        )  # saves png of each frame as "frame001.png, frame002.png, .."

        # stop pymol instance
        if self.pse or self.pse_share:
            logger.info("Writing PyMol ensemble to session_6_final.pse...")
            p.cmd.save(str(parent_path / "session_6_final.pse"))
        p.stop()

        # TODO: higher resolution on the pngs.
        # TODO: Find way to improve writing speed by e.g. removing atoms not in view. Currently takes ~80sec per .png
        # use imagio to create a gif from the .png files generated by pymol

        from glob import glob

        import imageio.v2 as iio

        logger.info(f"Creating animated GIF {path} from images...")
        png_files = glob(f"{prefix}*.png")
        if len(png_files) == 0:
            raise OSError(f"No {prefix}*.png files found - did PyMol not generate any?")
        png_files.sort()  # for some reason *sometimes* this list is scrambled messing up the GIF. Sorting fixes the issue.

        # add progress bar to each frame

        add_gif_progress_bar(
            png_files, frames_per_ns=self.frames_per_ns, start_frame=self.start
        )

        with iio.get_writer(str(path), mode="I") as writer:
            for filename in png_files:
                image = iio.imread(filename)
                writer.append_data(image)

        # now compress the GIF with the method that imagio recommends (https://imageio.readthedocs.io/en/stable/examples.html).
        logger.info("Compressing animated gif...")
        gifsicle(
            sources=str(path),  # happens in-place
            optimize=True,
            colors=256,
            options=["--loopcount"],  # this makes sure the GIF loops
        )

        # remove tmpdir
        shutil.rmtree(tmpdir)

        return path


def add_gif_progress_bar(
    png_files: list[Union[Path, str]], frames_per_ns: int, start_frame: int = 1
) -> None:
    """
    adds a progress bar and nanosecond counter onto PNG images. This assumes PNG
    files are named with index in the form path/frame<INDEX>.png. Overlaying of these objects
    happens in-place.

    Parameters
    ----------
    png_files : List[Union[Path, str]]
        List of PNG paths to add progress bars to.
    frames_per_ns : int
        Number of frames per nanosecond
    start_frame : int
        Frame to start from. Default is 1, which is the first frame, note indexed at 1.
    """
    from PIL import Image, ImageDraw, ImageFont

    # global settings:
    total_frames = len(png_files)

    for filename in png_files:
        filename = str(filename)
        # get this file's frame number from the filename and calculate total amount of ns simulated for this frame
        frame_num = int(filename.split("frame")[1].split(".png")[0])
        # adjust for the fact that we may not have started at the first frame, which will still be written out  as frame0001.png
        # note 1 indexing here.
        frame_num_actual = frame_num + start_frame - 1
        total_ns_this_frame = f"{frame_num_actual / frames_per_ns:.3f}"

        # load the image.
        img = Image.open(filename)
        img2 = Image.new("RGBA", img.size, "WHITE")
        img2.paste(img, mask=img)
        draw = ImageDraw.Draw(img2, "RGBA")

        # get its dimensions (need these for coords); calculate progress bar width at this frame.
        width, height = img2.size
        bar_width = frame_num / total_frames * width

        # draw the progress bar for this frame (black, fully opaque).
        draw.rectangle(((0, height - 20), (bar_width, height)), fill=(0, 0, 0, 500))
        # draw the text that shows time progression.
        draw.text(
            (width - 125, height - 10),
            f"{total_ns_this_frame} ns",
            # need to load a local font. For some odd reason this is the only way to write text with PIL.
            font=ImageFont.truetype(opensans_regular, 65),
            fill=(0, 0, 0),  # make all black.
            anchor="md",
            stroke_width=2,
            stroke_fill="white",
        )  # align to RHS; this way if value increases it will grow into frame.

        # save the image.
        img2.save(filename)
