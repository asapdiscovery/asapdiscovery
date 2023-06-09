import logging
import shutil
from pathlib import Path
from typing import List, Optional, Union  # noqa: F401

from asapdiscovery.data.logging import FileLogger

from ._gif_blocks import (
    color_dict,
    pocket_dict_mers,
    pocket_dict_sars2,
    view_coord_mers,
    view_coords_7ene,
    view_coords_272,
    view_coords_sars2,
)
from .show_contacts import show_contacts
from .resources.fonts import opensans_regular

class GIFVisualizer:
    """
    Class for generating GIF visualizations of MD trajectories.
    """

    allowed_targets = (
        "sars2",
        "mers",
        "7ene",
        "272",
    )

    # TODO: replace input with a schema rather than paths.
    def __init__(
        self,
        trajectories: list[Path],
        systems: list[Path],
        output_paths: list[Path],
        target: str,
        frames_per_ns: int = 200,
        pse: bool = False,
        pse_share: bool = True,
        smooth: int = 0,
        contacts: bool = True,
        start: int = 1,
        stop: int = -1,
        interval: int = 1,
        logger: FileLogger = None,
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
            Target to visualize poses for. Must be one of: "sars2", "mers", "7ene", "272".
        pse : bool
            Whether to write PyMol session files.
        smooth : int
            Number of frames to smooth over.
        contacts : bool
            Whether to show contacts.
        start : int
            Start frame to load
        stop : int
            Stop frame to load
        interval : int
            Interval between frames to load
        logger : FileLogger
            Logger to use
        debug : bool
            Whether to run in debug mode.

        """
        if not len(trajectories) == len(output_paths):
            raise ValueError("Number of trajectories and paths must be equal.")

        # init logger
        if logger is None:
            self.logger = FileLogger(
                "gif_visualizer_log.txt", "./", stdout=True, level=logging.INFO
            ).getLogger()
        else:
            self.logger = logger

        if target not in self.allowed_targets:
            raise ValueError(f"Target must be one of: {self.allowed_targets}")
        self.target = target
        self.logger.info(f"Visualizing trajectories for {self.target}")

        # setup pocket dict and view_coords for target
        if self.target == "sars2":
            self.pocket_dict = pocket_dict_sars2
            self.view_coords = view_coords_sars2
        elif self.target == "mers":
            self.pocket_dict = pocket_dict_mers
            self.view_coords = view_coord_mers
        elif self.target == "7ene":
            self.pocket_dict = pocket_dict_sars2
            self.view_coords = view_coords_7ene
        elif self.target == "272":
            self.pocket_dict = pocket_dict_mers
            self.view_coords = view_coords_272

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
            else:
                self.logger.warning(
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

        self.debug = debug
        if self.debug:
            self.logger.SetLevel(logging.DEBUG)
            self.logger.debug("Running in debug mode, setting pse=True")
            self.pse = True
        self.logger.debug(
            f"Writing GIF visualizations for {len(self.output_paths)} ligands"
        )

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
        self.logger.info(f"Creating temporary directory {tmpdir}")
        tmpdir.mkdir(parents=True, exist_ok=True)

        complex_name = "complex"
        p.cmd.load(str(system), object=complex_name)

        if self.pse:
            self.logger.info("Writing PyMol ensemble to session_1_loaded_system.pse...")
            p.cmd.save(str(parent_path / "session_1_loaded_system.pse"))

        # now select the residues, name them and color them.
        for subpocket_name, residues in self.pocket_dict.items():
            p.cmd.select(
                subpocket_name,
                f"{complex_name} and resi {residues} and polymer.protein",
            )

        for subpocket_name, color in color_dict.items():
            p.cmd.set("surface_color", color, f"({subpocket_name})")

        if self.pse:
            self.logger.info(
                "Writing PyMol ensemble to session_2_colored_subpockets.pse"
            )
            p.cmd.save(str(parent_path / "session_2_colored_subpockets.pse"))

        # Select ligand and receptor
        p.cmd.select("ligand", "resn UNK")
        p.cmd.select(
            "receptor", "chain A or chain B"
        )  # TODO: Modify this to generalize to dimer
        p.cmd.select(
            "binding_site", "name CA within 7 of resn UNK"
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
        p.cmd.select("resn UNK")
        p.cmd.show("sticks", "sele")
        p.cmd.show("sticks", "subP*")
        p.cmd.hide("sticks", "(elem C extend 1) and (elem H)")
        p.cmd.color("pink", "elem C and sele")

        p.cmd.set_view(self.view_coords)
        if self.pse or self.pse_share:
            p.cmd.save(str(parent_path / "session_3_set_ligand_view.pse"))

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

        self.logger.info("Intrafitting simulation...")
        p.cmd.intra_fit("binding_site")
        if self.smooth:
            p.cmd.smooth(
                "all", window=int(self.smooth)
            )  # perform some smoothing of frames
        p.cmd.zoom("resn UNK", buffer=1)  # zoom to ligand

        if self.contacts:
            self.logger.info("Showing contacts...")
            show_contacts(p, "ligand", "receptor")

        if self.pse:
            self.logger.info("Writing PyMol ensemble to session_5_intrafitted.pse...")
            p.cmd.save(str(parent_path / "session_5_intrafitted.pse"))

        # Process the trajectory in a temporary directory
        from pygifsicle import optimize

        # now make the movie.
        self.logger.info("Rendering images for frames...")
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
        p.stop()

        # TODO: higher resolution on the pngs.
        # TODO: Find way to improve writing speed by e.g. removing atoms not in view. Currently takes ~80sec per .png
        # use imagio to create a gif from the .png files generated by pymol

        from glob import glob

        import imageio.v2 as iio

        self.logger.info(f"Creating animated GIF {path} from images...")
        png_files = glob(f"{prefix}*.png")
        if len(png_files) == 0:
            raise OSError(f"No {prefix}*.png files found - did PyMol not generate any?")
        png_files.sort()  # for some reason *sometimes* this list is scrambled messing up the GIF. Sorting fixes the issue.

        # add progress bar to each frame

        add_gif_progress_bar(png_files, frames_per_ns=self.frames_per_ns)

        with iio.get_writer(str(path), mode="I") as writer:
            for filename in png_files:
                image = iio.imread(filename)
                writer.append_data(image)

        # now compress the GIF with the method that imagio recommends (https://imageio.readthedocs.io/en/stable/examples.html).
        self.logger.info("Compressing animated gif...")
        optimize(str(path))  # this is in-place.

        # remove tmpdir
        shutil.rmtree(tmpdir)

        return path


def add_gif_progress_bar(png_files: list[Union[Path, str]], frames_per_ns: int) -> None:
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
    """
    from PIL import Image, ImageDraw, ImageFont

    # global settings:
    total_frames = len(png_files)

    for filename in png_files:
        filename = str(filename)
        # get this file's frame number from the filename and calculate total amount of ns simulated for this frame
        frame_num = int(filename.split("frame")[1].split(".png")[0])
        total_ns_this_frame = f"{frame_num / frames_per_ns:.3f}"

        # load the image.
        img = Image.open(filename)
        draw = ImageDraw.Draw(img, "RGBA")

        # get its dimensions (need these for coords); calculate progress bar width at this frame.
        width, height = img.size
        bar_width = frame_num / total_frames * width

        # draw the progress bar for this frame (black, fully opaque).
        draw.rectangle(((0, height - 20), (bar_width, height)), fill=(0, 0, 0, 500))

        # draw the text that shows
        draw.text(
            (width - 110, height - 10),
            f"{total_ns_this_frame} ns",
            # need to load a local font. For some odd reason this is the only way to write text with PIL.
            font=ImageFont.truetype(opensans_regular, 65),
            fill=(0, 0, 0),  # make all black.
            anchor="md",
        )  # align to RHS; this way if value increases it will grow into frame.

        # save the image.
        img.save(filename)
