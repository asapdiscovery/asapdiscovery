import argparse
import logging
from pathlib import Path

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
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
    "--pose",
    type=str,
    required=True,
    help="Path to the pose SDF file",
)

parser.add_argument(
    "--protein",
    type=str,
    required=True,
    help="Path to the protein file",
)

parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Path to the output HTML file",
)

def main():
    args = parser.parse_args()

    # setup logging
    logger_cls = FileLogger("pose_to_viz", path="./", stdout=True, level=logging.DEBUG)
    logger = logger_cls.getLogger()
    logger.info("Running HTML visualization")

    # check all the required files exist
    pose = Path(args.pose)
    if not pose.exists():
        raise FileNotFoundError(f"Pose file {pose} does not exist")
    protein = Path(args.protein)
    if not protein.exists():
        raise FileNotFoundError(f"Topology file {protein} does not exist")
    
    out = Path(args.out)

    logger.info(f"Pose file: {pose}")
    logger.info(f"Protein file: {protein}")
    logger.info(f"Output file: {out}")

    html_visualizer = HTMLVisualizer(
        poses=[str(pose)],
        output_paths=[out],
        target=args.viz_target,
        protein= protein,
        logger=logger,
        
    )
    html_visualizer.write_pose_visualizations()

    logger.info("Done")


if __name__ == "__main__":
    main()
