import argparse
import logging
from pathlib import Path

import mdtraj as md
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
from asapdiscovery.dataviz.viz_targets import VizTargets

parser = argparse.ArgumentParser(description="Turn a trajectory into a GIF")

parser.add_argument(
    "--viz-target",
    type=str,
    required=True,
    choices=VizTargets.get_allowed_targets(),
    help="Target to write visualizations for, one of (sars2_mpro, mers_mpro, 7ene_mpro, 272_mpro, sars2_mac1)",
)

parser.add_argument(
    "--traj",
    type=str,
    required=True,
    help="Path to the trajectory to visualize",
)

parser.add_argument(
    "--top",
    type=str,
    required=True,
    help="Path to the topology file",
)

parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Path to the output gif file",
)

parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="Frame to start visualization from",
)

parser.add_argument(
    "--frames_per_ns",
    type=int,
    default=200,
    help="Number of frames per ns, default matches the default output frequency for VanillaMDSimulator",
)

parser.add_argument(
    "--smooth",
    type=int,
    default=5,
    help="Number of frames to smooth over",
)


def main():
    args = parser.parse_args()

    # setup logging
    logger_cls = FileLogger("traj_to_viz", path="./", stdout=True, level=logging.DEBUG)
    logger = logger_cls.getLogger()
    logger.info("Running GIF visualization")

    # check all the required files exist
    traj = Path(args.traj)
    if not traj.exists():
        raise FileNotFoundError(f"Trajectory file {traj} does not exist")
    top = Path(args.top)
    if not top.exists():
        raise FileNotFoundError(f"Topology file {top} does not exist")
    out = Path(args.out)

    logger.info(f"Trajectory file: {traj}")
    logger.info(f"Topology file: {top}")
    logger.info(f"Output file: {out}")

    logger.info("Loading trajectory")
    _traj = md.load(str(traj), top=str(top))
    n_snapshots = _traj.n_frames
    logger.info(f"Loaded {n_snapshots} snapshots")

    if not args.start:
        # we want the last 100 snapshots
        if n_snapshots < 100:
            start = 1
        else:
            start = n_snapshots - 100
    else:
        start = args.start

    logger.info(f"Starting from snapshot {start}")

    gif_visualiser = GIFVisualizer(
        [traj],
        [args.top],
        [out],
        args.viz_target,
        frames_per_ns=args.frames_per_ns,
        smooth=args.smooth,
        start=start,
        logger=logger,
    )
    gif_visualiser.write_traj_visualizations()

    logger.info("Done")


if __name__ == "__main__":
    main()
