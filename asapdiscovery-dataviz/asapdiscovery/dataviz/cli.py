import logging
from pathlib import Path
from shutil import rmtree
from typing import Union

import click
import mdtraj as md
from asapdiscovery.cli.cli_args import (
    ligands,
    loglevel,
    output_dir,
    pdb_file,
    target,
    use_dask,
)
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.util.dask_utils import DaskType, make_dask_client_meta
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
from asapdiscovery.dataviz.html_viz import HTMLVisualizer


@click.group()
def visualization():
    """Create vizualizations on molecular systems."""
    pass


@visualization.command()
@click.option(
    "--colour-method",
    default="subpockets",
    help="Coloring method",
    type=click.Choice(["subpockets", "fitness"]),
)
@click.option("--align", is_flag=True, help="Align the protein to reference structure")
@target
@ligands
@pdb_file
@output_dir
@loglevel
@use_dask
def pose_html(
    colour_method: str,
    align: bool,
    target: TargetTags,
    ligands: str,
    pdb_file: str,
    output_dir: str = "poses",
    loglevel: Union[int, str] = logging.INFO,
    use_dask: bool = False,
):
    """Create a HTML file with the pose of the ligand in the protein."""
    # make output directory
    output_dir = Path(output_dir)

    if output_dir.exists():
        rmtree(output_dir)

    output_dir.mkdir()

    logger = FileLogger(
        "",  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="viz.log",
        stdout=True,
        level=loglevel,
    ).getLogger()
    logger.info("Running HTML visualization")

    if use_dask:
        logger.info("Using dask")
        dask_client = make_dask_client_meta(DaskType.LOCAL, loglevel=loglevel)
    else:
        dask_client = None

    # check all the required files exist
    ligands = Path(ligands)
    if not ligands.exists():
        raise FileNotFoundError(f"Ligand file {ligands} does not exist")
    protein = Path(pdb_file)
    if not protein.exists():
        raise FileNotFoundError(f"Topology file {protein} does not exist")

    logger.info(f"Ligand file: {ligands}")
    logger.info(f"Protein file: {protein}")
    logger.info(f"Output directory: {output_dir}")

    html_visualizer = HTMLVisualizer(
        color_method=colour_method,
        target=target,
        align=align,
        write_to_disk=True,
        output_dir=output_dir,
    )
    ligs = MolFileFactory(filename=ligands).load()
    cmplx = Complex.from_pdb(
        protein,
        target_kwargs={"target_name": protein.stem},
        ligand_kwargs={"compound_name": protein.stem + "_ligand"},
    )

    cmplx.to_pdb(output_dir / "protein.pdb")

    html_visualizer.visualize(
        inputs=[(cmplx, ligs)], use_dask=use_dask, dask_client=dask_client
    )
    logger.info("Done")


@visualization.command()
@click.option("--traj", required=True, help="Path to the trajectory file.")
@click.option("--top", required=True, help="Path to the topology file.")
@output_dir
@click.option(
    "--start",
    default=0,
    type=int,
    help="Starting snapshot. Defaults to last 100 snapshots if not specified.",
)
@target
@click.option(
    "--frames_per_ns",
    default=200,
    type=int,
    help="Frames per nanosecond, default matches the default output frequency for VanillaMDSimulator",
)
@click.option("--smooth", default=5, type=int, help="Number of frames to smooth over")
@click.option(
    "--pymol-debug",
    is_flag=True,
    help="PyMOL debugging, will produce pymol sessions rather than a GIF",
)
def traj_gif(
    traj: str,
    top: str,
    output_dir: str,
    start: int,
    target: TargetTags,
    frames_per_ns: int,
    smooth: int,
    pymol_debug: bool,
    loglevel: Union[int, str] = logging.INFO,
):
    """Create a GIF from a trajectory."""

    # make output directory
    output_dir = Path(output_dir)

    if output_dir.exists():
        rmtree(output_dir)

    output_dir.mkdir()

    logger = FileLogger(
        "",  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="gif_viz.log",
        stdout=True,
        level=loglevel,
    ).getLogger()

    logger.info("Running GIF visualization")

    traj_path = Path(traj)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file {traj_path} does not exist")
    top_path = Path(top)
    if not top_path.exists():
        raise FileNotFoundError(f"Topology file {top_path} does not exist")

    logger.info(f"Trajectory file: {traj_path}")
    logger.info(f"Topology file: {top_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target: {target}")
    logger.info(f"Frames per nanosecond: {frames_per_ns}")

    logger.info("Loading trajectory")
    _traj = md.load(str(traj_path), top=str(top_path))
    n_snapshots = _traj.n_frames
    logger.info(f"Loaded {n_snapshots} snapshots")

    if not start:
        logger.info("Start not specified, using last 100 snapshots")
        if n_snapshots < 100:
            start = 1
        else:
            start = n_snapshots - 99

    logger.info(f"Starting from snapshot {start}")

    if pymol_debug:
        gif_visualiser = GIFVisualizer(
            target=target,
            frames_per_ns=5,  # dummy value
            smooth=5,
            static_view_only=True,
            start=1,  # dummy value
            pse=True,  # can set these to True to debug viz steps.
            pse_share=False,
            output_dir=output_dir,
        )
        gif_visualiser.visualize(inputs=[(None, top_path)])

    else:
        gif_visualiser = GIFVisualizer(
            target=target,
            frames_per_ns=frames_per_ns,
            smooth=smooth,
            start=start,
            output_dir=output_dir,
        )
        gif_visualiser.visualize(
            inputs=[(traj_path, top_path)], outpaths=[output_dir / "traj.gif"]
        )

    logger.info("Done")
