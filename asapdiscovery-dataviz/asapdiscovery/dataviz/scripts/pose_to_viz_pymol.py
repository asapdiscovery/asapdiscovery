import argparse
import logging
from pathlib import Path

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.dataviz.gif_viz import GIFVisualizer

parser = argparse.ArgumentParser(
    description="Turn a PDB complex into a PyMOL PSE file with canonical target view"
)

parser.add_argument(
    "--target",
    type=str,
    required=True,
    choices=TargetTags.get_values(),
    help="Target to write visualizations for",
)

parser.add_argument(
    "--complex",
    type=str,
    required=True,
    help="Path to the PDB ligand-protein complex file (UNK or LIG as ligand entry in PDB file)",
)

parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Path to the output .PSE file",
)


def main():
    args = parser.parse_args()

    # setup logging
    logger_cls = FileLogger(
        "pose_to_viz_pymol", path="./", stdout=True, level=logging.DEBUG
    )
    logger = logger_cls.getLogger()
    logger.info("Running pose visualization")

    # check all the required files exist
    complex = Path(args.complex)
    if not complex.exists():
        raise FileNotFoundError(f"Topology file {complex} does not exist")
    out = Path(args.out)
    logger.info(f"Output file: {out}")

    gif_visualiser = GIFVisualizer(
        [None],  # we just fill these args, they're not being used.
        [complex],
        [out],
        args.viz_target,
        frames_per_ns=0,
        smooth=5,
        static_view_only=True,
        start=0,
        pse=False,  # can set these to True to debug viz steps.
        pse_share=False,
    )
    gif_visualiser.write_traj_visualization(traj=None, system=complex, path=out)

    logger.info("Done")


if __name__ == "__main__":
    main()
