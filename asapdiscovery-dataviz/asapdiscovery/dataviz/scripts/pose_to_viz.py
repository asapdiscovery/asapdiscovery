import argparse
import logging
from pathlib import Path

from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.schema.complex import Complex

parser = argparse.ArgumentParser(description="Turn a trajectory into a GIF")

parser.add_argument(
    "--target",
    type=str,
    required=True,
    choices=TargetTags.get_values(),
    help="Target to write visualizations for",
)

parser.add_argument(
    "--pose",
    type=str,
    required=True,
    help="Path to the pose file, can be an SDF or MOL2 file",
)

parser.add_argument(
    "--protein",
    type=str,
    required=True,
    help="Path to the protein file",
)

parser.add_argument(
    "--color-method",
    type=str,
    required=False,
    default="subpockets",
    choices=["subpockets", "fitness"],
    help="Whether to show subpocket color mapping (default) or fitness color mapping",
)


parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Path to the output HTML file",
)

parser.add_argument(
    "--align",
    action="store_true",
    help="Align the protein to canoncical target view",
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
        color_method=args.color_method,
        target=args.target,
        align=args.align,
        write_to_disk=True,
        output_dir="./",
    )
    ligs = MolFileFactory(filename=pose).load()
    cmplx = Complex.from_pdb(
        protein,
        target_kwargs={"target_name": "unknown_target"},
        ligand_kwargs={"compound_name": f"unknown_compound"},
    )

    html_visualizer.visualize(inputs=[(cmplx, ligs)], outpaths=[out])

    logger.info("Done")


if __name__ == "__main__":
    main()
