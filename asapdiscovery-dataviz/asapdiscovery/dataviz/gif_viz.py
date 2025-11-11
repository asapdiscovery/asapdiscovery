import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Union

from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.data.services.postera.manifold_data_validation import (
    TargetProteinMap,
    TargetTags,
)
from asapdiscovery.data.util.dask_utils import backend_wrapper, dask_vmap
from asapdiscovery.dataviz._gif_blocks import GIFBlockData
from asapdiscovery.dataviz.resources.fonts import opensans_regular
from asapdiscovery.dataviz.show_contacts import show_contacts
from asapdiscovery.dataviz.visualizer import VisualizerBase
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.simulation.simulate import SimulationResult
from multimethod import multimethod
from pydantic.v1 import Field, PositiveInt

logger = logging.getLogger(__name__)


class GIFVisualizer(VisualizerBase):
    """
    Class for generating GIF visualizations of MD simulations.

    The GIFVisualizer class is used to generate GIF visualizations of MD simulations.

    The main method is `visualize`, which can accept a list of inputs and an optional list of output paths.
    The `visualize` method is overloaded to accept either a list of SimulationResult objects or a list of tuples of paths to trajectory and topology files.
    Optionally if using the `Path` overload, the trajectory member of the tuple can be `None` if generating a static view (static_view_only=True).


    Parameters
    ----------
    target : TargetTags
        Target to visualize poses for
    debug : bool
        Whether to run in debug mode
    output_dir : Path
        Output directory to write HTML files to
    frames_per_ns : PositiveInt
        Number of frames per ns
    pse : bool
        Whether to generate PSE files
    pse_share : bool
        Whether to generate PSE files with shared view
    smooth : PositiveInt
        How many frames over which to smooth the trajectory
    contacts : bool
        Whether to generate contact maps
    static_view_only : bool
        Whether to only generate static PSE for the trajectory
    zoom_view : bool
        Whether to zoom into the binding site on final output visualization
    start : PositiveInt
        Start frame - if not defined, will default to last 10% of default trajectory settings
    stop : int
        Stop frame
    interval : PositiveInt
        Interval between frames
    """

    target: TargetTags = Field(..., description="Target to visualize poses for")
    debug: bool = Field(False, description="Whether to run in debug mode")
    output_dir: Path = Field(
        "gif", description="Output directory to write HTML files to"
    )
    frames_per_ns: PositiveInt = Field(200, description="Number of frames per ns")
    pse: bool = Field(False, description="Whether to generate PSE files")
    pse_share: bool = Field(
        False, description="Whether to generate PSE files with shared view"
    )
    smooth: PositiveInt = Field(
        3, description="How many frames over which to smooth the trajectory"
    )
    contacts: bool = Field(True, description="Whether to generate contact maps")
    static_view_only: bool = Field(
        False, description="Whether to only generate static views"
    )
    zoom_view: bool = Field(
        False,
        description="Whether to zoom into the binding site on final output visualization",
    )
    start: PositiveInt = Field(
        1800,
        description="Start frame - if not defined, will default to last 10% of default trajectory settings",
    )
    stop: int = Field(-1, description="Stop frame")
    interval: PositiveInt = Field(1, description="Interval between frames")
    debug: bool = Field(False, description="Whether to run in debug mode")

    class Config:
        arbitrary_types_allowed = True

    @dask_vmap(["inputs"], has_failure_mode=True)
    @backend_wrapper("inputs")
    def _visualize(
        self, inputs: list[Any], outpaths: Optional[list[Path]] = None, **kwargs
    ) -> list[dict[str, str]]:
        if outpaths:
            if len(outpaths) != len(inputs):
                raise ValueError("outpaths must be the same length as inputs")

        return self._dispatch(inputs, outpaths=outpaths, **kwargs)

    def provenance(self):
        return {}

    @staticmethod
    def pymol_traj_viz(
        target: str,
        traj: Optional[Path],
        system: Path,
        static_view_only: bool,
        pse: bool,
        pse_share: bool,
        start: int,
        stop: int,
        interval: int,
        smooth: int,
        contacts: bool,
        frames_per_ns: int,
        zoom_view: bool,
        outpath: Optional[Path] = None,
        out_dir: Optional[Path] = None,
    ):
        view_coords = GIFBlockData.get_view_coords()

        pocket_dict = GIFBlockData.get_pocket_dict(target)

        color_dict = GIFBlockData.get_color_dict(target)

        if not out_dir and not outpath:
            raise ValueError("Either out_dir or outpath must be defined")

        import pymol2

        p = pymol2.PyMOL()
        p.start()
        if outpath:
            out_dir = Path(outpath).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            path = outpath
        else:
            if static_view_only:
                path = out_dir / "view.pse"
            else:
                path = out_dir / "trajectory.gif"

        # get path without filename
        out_dir = Path(outpath).parent
        tmpdir = out_dir / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        complex_name = "complex"
        p.cmd.load(str(system), object=complex_name)
        if static_view_only:
            # this may be unprepped/unaligned, so need to align to master structure before writing out.
            reference_structure = master_structures[target]
            p.cmd.load(reference_structure, object="reference_master")
            p.cmd.align(complex_name, "reference_master")
            p.cmd.delete("reference_master")

        if pse:
            p.cmd.save(str(out_dir / "session_1_loaded_system.pse"))

        # now select the residues, name them and color them.
        for subpocket_name, residues in pocket_dict.items():
            p.cmd.select(
                subpocket_name,
                f"{complex_name} and resi {residues} and polymer.protein",
            )

        for subpocket_name, color in color_dict.items():
            p.cmd.set("surface_color", color, f"({subpocket_name})")

        if pse:
            p.cmd.save(str(out_dir / "session_2_colored_subpockets.pse"))

        # Select ligand and receptor
        p.cmd.select("ligand", "resn UNK or resn LIG")
        p.cmd.select(
            "receptor", "chain A or chain B or chain 1 or chain 2"
        )  # TODO: Modify this to generalize to dimer
        p.cmd.select(
            "binding_site",
            "name CA within 7 of resn UNK or name CA within 7 resn LIG",
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

        for subpocket_name, color in color_dict.items():
            # set non-polar sticks for this subpocket, color the backbone by subpocket color.
            p.cmd.select(subpocket_name)
            p.cmd.show("sticks", "sele")
            p.cmd.set("stick_color", color, f"({subpocket_name})")
            p.cmd.hide("sticks", "(elem C extend 1) and (elem H)")

        if pse:
            p.cmd.save(str(out_dir / "session_3_set_ligand_view.pse"))

        if not static_view_only:
            # load trajectory; center the system in the simulation and smoothen between frames.
            p.cmd.load_traj(
                str(traj),
                object=complex_name,
                start=start,
                stop=stop,
                interval=interval,
            )
            # fix pbc
            p.cmd.pbc_wrap(complex_name)

            if pse:
                p.cmd.save(str(out_dir / "session_4_loaded_trajectory.pse"))

            # center the system to the minimized structure
            # reload
            complex_name_min = "complex_min"
            p.cmd.load(str(system), object=complex_name_min)
            # align to reference structure
            reference_structure = master_structures[target]
            p.cmd.load(reference_structure, object="reference_master")
            p.cmd.align(complex_name_min, "reference_master")
            p.cmd.delete("reference_master")

            # align the trajectory to the minimized structure (itself aligned to the reference structure)
            p.cmd.align(complex_name, complex_name_min)
            p.cmd.intra_fit(complex_name, 1)
            p.cmd.delete(complex_name_min)

            if smooth:
                p.cmd.smooth(
                    "all", window=int(smooth)
                )  # perform some smoothing of frames

        if contacts:
            p.cmd.extract("ligand_obj", "ligand")

            show_contacts(p, "ligand_obj", "receptor")

        p.cmd.set_view(view_coords)  # sets general orientation
        if static_view_only:
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
        if TargetProteinMap[target] == "Capsid":
            p.cmd.clip("near", -25)

        if pse or pse_share:
            p.cmd.save(str(out_dir / "session_5_selections.pse"))

        # Process the trajectory in a temporary directory
        from pygifsicle import gifsicle

        if zoom_view:
            p.cmd.zoom("binding_site")
        # now make the movie.
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
        if pse or pse_share:
            p.cmd.save(str(out_dir / "session_6_final.pse"))
        p.stop()

        from glob import glob

        import imageio.v2 as iio

        png_files = glob(f"{prefix}*.png")
        if len(png_files) == 0:
            raise OSError(f"No {prefix}*.png files found - did PyMol not generate any?")
        png_files.sort()  # for some reason *sometimes* this list is scrambled messing up the GIF. Sorting fixes the issue.

        # add progress bar to each frame

        add_gif_progress_bar(png_files, frames_per_ns=frames_per_ns, start_frame=start)

        with iio.get_writer(str(path), mode="I") as writer:
            for filename in png_files:
                image = iio.imread(filename)
                writer.append_data(image)

        # now compress the GIF with the method that imagio recommends (https://imageio.readthedocs.io/en/stable/examples.html).
        gifsicle(
            sources=str(path),  # happens in-place
            optimize=True,
            colors=256,
            options=["--loopcount"],  # this makes sure the GIF loops
        )

        # remove tmpdir
        shutil.rmtree(tmpdir)

        return path

    @multimethod
    def _dispatch(
        self,
        inputs: list[SimulationResult],
        outpaths: Optional[list[Path]] = None,
        failure_mode: str = "skip",
        **kwargs,
    ):
        data = []

        for i, res in enumerate(inputs):
            try:
                if not outpaths:
                    if self.static_view_only:
                        out = (
                            self.output_dir
                            / res.input_docking_result.unique_name
                            / "view.pse"
                        )
                    else:
                        out = (
                            self.output_dir
                            / res.input_docking_result.unique_name
                            / "trajectory.gif"
                        )
                else:
                    out = self.output_dir / outpaths[i]

                path = self.pymol_traj_viz(
                    target=self.target,
                    traj=res.traj_path,
                    system=res.minimized_pdb_path,
                    outpath=out,
                    static_view_only=self.static_view_only,
                    pse=self.pse,
                    pse_share=self.pse_share,
                    start=self.start,
                    stop=self.stop,
                    interval=self.interval,
                    smooth=self.smooth,
                    contacts=self.contacts,
                    frames_per_ns=self.frames_per_ns,
                    zoom_view=self.zoom_view,
                    out_dir=self.output_dir,
                )
                row = {}
                row[DockingResultCols.LIGAND_ID.value] = (
                    res.input_docking_result.posed_ligand.compound_name
                )
                row[DockingResultCols.TARGET_ID.value] = (
                    res.input_docking_result.input_pair.complex.target.target_name
                )
                row[DockingResultCols.SMILES.value] = (
                    res.input_docking_result.posed_ligand.smiles
                )
                row[DockingResultCols.GIF_PATH.value] = path
                data.append(row)
            except Exception as e:
                if failure_mode == "skip":
                    logger.error(
                        f"Error processing {res.input_docking_result.unique_name}: {e}"
                    )
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )
        return data

    @_dispatch.register
    def _dispatch(
        self,
        inputs: list[tuple[Optional[Path], Path]],
        outpaths: Optional[list[Path]] = None,
        failure_mode: str = "skip",
        **kwargs,
    ):
        data = []
        for i, tup in enumerate(inputs):
            try:
                # unpack the tuple
                traj, top = tup
                # find with the unique identifier for the ligand would be
                complex = Complex.from_pdb(
                    top,
                    target_kwargs={"target_name": f"{top.stem}_target"},
                    ligand_kwargs={"compound_name": f"{top.stem}_ligand"},
                )
                csp = CompoundStructurePair(complex=complex, ligand=complex.ligand)
                if not outpaths:
                    if self.static_view_only:
                        out = self.output_dir / csp.unique_name / "view.pse"
                    else:
                        out = self.output_dir / csp.unique_name / "trajectory.gif"
                else:
                    out = self.output_dir / outpaths[i]

                path = self.pymol_traj_viz(
                    target=self.target,
                    traj=traj,
                    system=top,
                    outpath=out,
                    static_view_only=self.static_view_only,
                    pse=self.pse,
                    pse_share=self.pse_share,
                    start=self.start,
                    stop=self.stop,
                    interval=self.interval,
                    smooth=self.smooth,
                    contacts=self.contacts,
                    frames_per_ns=self.frames_per_ns,
                    zoom_view=self.zoom_view,
                    out_dir=self.output_dir,
                )
                row = {}
                row[DockingResultCols.LIGAND_ID.value] = complex.ligand.compound_name
                row[DockingResultCols.TARGET_ID.value] = complex.target.target_name
                row[DockingResultCols.SMILES.value] = complex.ligand.smiles
                row[DockingResultCols.GIF_PATH.value] = path
                data.append(row)
            except Exception as e:
                if failure_mode == "skip":
                    logger.error(f"Error processing {csp.unique_name}: {e}")
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )
        return data


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
