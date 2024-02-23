import logging
import shutil
from pathlib import Path

import pandas as pd
from asapdiscovery.data.util.dask_utils import dask_vmap
from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.services.postera.manifold_data_validation import (
    TargetProteinMap,
    TargetTags,
)
from asapdiscovery.data.util.dask_utils import (
    DaskFailureMode,
    actualise_dask_delayed_iterable,
)
from asapdiscovery.dataviz._gif_blocks import GIFBlockData
from asapdiscovery.dataviz.gif_viz import add_gif_progress_bar
from asapdiscovery.dataviz.show_contacts import show_contacts
from asapdiscovery.dataviz.viz_v2.visualizer import VisualizerBase
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.simulation.simulate import SimulationResult
from pydantic import Field, PositiveInt

logger = logging.getLogger(__name__)


class GIFVisualizerV2(VisualizerBase):
    """
    Class for generating GIF visualizations of MD simulations.
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
    start: PositiveInt = Field(
        1800,
        description="Start frame - if not defined, will default to last 10% of default trajectory settings",
    )
    stop: int = Field(-1, description="Stop frame")
    interval: PositiveInt = Field(1, description="Interval between frames")
    debug: bool = Field(False, description="Whether to run in debug mode")

    class Config:
        arbitrary_types_allowed = True

    @dask_vmap(["inputs"])
    def _visualize(
        self, inputs: list[SimulationResult], **kwargs
    ) -> list[dict[str, str]]:
        view_coords = GIFBlockData.get_view_coords()

        pocket_dict = GIFBlockData.get_pocket_dict(self.target)

        color_dict = GIFBlockData.get_color_dict(self.target)

        data = []
        for res in inputs:
            traj = res.traj_path
            system = res.minimized_pdb_path
            # NOTE very important, need to spawn a new pymol proc for each trajectory
            # when working in parallel, otherwise they will trip over each other and not work.

            import pymol2

            p = pymol2.PyMOL()
            p.start()

            out_dir = self.output_dir / res.input_docking_result.unique_name
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "trajectory.gif"

            tmpdir = out_dir / "tmp"
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
                p.cmd.save(str(out_dir / "session_1_loaded_system.pse"))

            # now select the residues, name them and color them.
            for subpocket_name, residues in pocket_dict.items():
                p.cmd.select(
                    subpocket_name,
                    f"{complex_name} and resi {residues} and polymer.protein",
                )

            for subpocket_name, color in color_dict.items():
                p.cmd.set("surface_color", color, f"({subpocket_name})")

            if self.pse:
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

            if self.pse:
                p.cmd.save(str(out_dir / "session_3_set_ligand_view.pse"))

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
                    p.cmd.save(str(out_dir / "session_4_loaded_trajectory.pse"))

                # center the system to the minimized structure
                # reload
                complex_name_min = "complex_min"
                p.cmd.load(str(system), object=complex_name_min)
                p.cmd.align(complex_name, complex_name_min)
                p.cmd.delete(complex_name_min)

                if self.smooth:
                    p.cmd.smooth(
                        "all", window=int(self.smooth)
                    )  # perform some smoothing of frames

            if self.contacts:
                p.cmd.extract("ligand_obj", "ligand")

                show_contacts(p, "ligand_obj", "receptor")

            p.cmd.set_view(view_coords)  # sets general orientation
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
                p.cmd.save(str(out_dir / "session_5_selections.pse"))

            # Process the trajectory in a temporary directory
            from pygifsicle import gifsicle

            # now make the movie.
            p.cmd.set(
                "ray_trace_frames", 0
            )  # ray tracing with surface representation is too expensive.
            p.cmd.set(
                "defer_builds_mode", 1
            )  # this saves memory for large trajectories
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
                p.cmd.save(str(out_dir / "session_6_final.pse"))
            p.stop()

            # TODO: higher resolution on the pngs.
            # TODO: Find way to improve writing speed by e.g. removing atoms not in view. Currently takes ~80sec per .png
            # use imagio to create a gif from the .png files generated by pymol

            from glob import glob

            import imageio.v2 as iio

            png_files = glob(f"{prefix}*.png")
            if len(png_files) == 0:
                raise OSError(
                    f"No {prefix}*.png files found - did PyMol not generate any?"
                )
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
            gifsicle(
                sources=str(path),  # happens in-place
                optimize=True,
                colors=256,
                options=["--loopcount"],  # this makes sure the GIF loops
            )

            # remove tmpdir
            shutil.rmtree(tmpdir)

            row = {}
            row[
                DockingResultCols.LIGAND_ID.value
            ] = res.input_docking_result.input_pair.ligand.compound_name
            row[
                DockingResultCols.TARGET_ID.value
            ] = res.input_docking_result.input_pair.complex.target.target_name
            row[
                DockingResultCols.SMILES.value
            ] = res.input_docking_result.input_pair.ligand.smiles
            row[DockingResultCols.GIF_PATH.value] = path
            data.append(row)
        return data

    def provenance(self):
        return {}
