import logging
import pytest

from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from asapdiscovery.dataviz.viz_targets import VizTargets


@pytest.fixture(scope="session")
def pose():
    pose = fetch_test_file("Mpro-P2660_0A_bound-prepped_ligand.sdf")
    return pose


@pytest.fixture(scope="session")
def protein():
    protein = fetch_test_file("Mpro-P2660_0A_bound-prepped_complex.pdb")
    return protein


# enumerate over the allowed targets, these will produce rubbish poses but
# that's fine for testing, mostly just testing that they will run
@pytest.mark.parametrize("target", VizTargets.get_allowed_targets())
@pytest.mark.parametrize(
    "logger",
    [
        None,
        FileLogger(
            "pose_to_viz", path="./", stdout=True, level=logging.DEBUG
        ).getLogger(),
    ],
)
def test_html_viz(pose, protein, logger, target, tmp_path):
    html_visualizer = HTMLVisualizer(
        poses=[pose],
        output_paths=[tmp_path / "html_viz.html"],
        target=target,
        protein=protein,
        logger=logger,
    )
    html_visualizer.write_pose_visualizations()
